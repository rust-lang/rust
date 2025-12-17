//! In certain situations, rust automatically inserts derefs as necessary: for
//! example, field accesses `foo.bar` still work when `foo` is actually a
//! reference to a type with the field `bar`. This is an approximation of the
//! logic in rustc (which lives in rustc_hir_analysis/check/autoderef.rs).

use std::fmt;

use hir_def::{TraitId, TypeAliasId};
use rustc_type_ir::inherent::{IntoKind, Ty as _};
use tracing::debug;

use crate::{
    ParamEnvAndCrate,
    db::HirDatabase,
    infer::InferenceContext,
    next_solver::{
        Canonical, DbInterner, ParamEnv, TraitRef, Ty, TyKind, TypingMode,
        infer::{
            DbInternerInferExt, InferCtxt,
            traits::{Obligation, ObligationCause, PredicateObligations},
        },
        obligation_ctxt::ObligationCtxt,
    },
};

const AUTODEREF_RECURSION_LIMIT: usize = 20;

/// Returns types that `ty` transitively dereferences to. This function is only meant to be used
/// outside `hir-ty`.
///
/// It is guaranteed that:
/// - the yielded types don't contain inference variables (but may contain `TyKind::Error`).
/// - a type won't be yielded more than once; in other words, the returned iterator will stop if it
///   detects a cycle in the deref chain.
pub fn autoderef<'db>(
    db: &'db dyn HirDatabase,
    env: ParamEnvAndCrate<'db>,
    ty: Canonical<'db, Ty<'db>>,
) -> impl Iterator<Item = Ty<'db>> + use<'db> {
    let interner = DbInterner::new_with(db, env.krate);
    let infcx = interner.infer_ctxt().build(TypingMode::PostAnalysis);
    let (ty, _) = infcx.instantiate_canonical(&ty);
    let autoderef = Autoderef::new(&infcx, env.param_env, ty);
    let mut v = Vec::new();
    for (ty, _steps) in autoderef {
        // `ty` may contain unresolved inference variables. Since there's no chance they would be
        // resolved, just replace with fallback type.
        let resolved = infcx.resolve_vars_if_possible(ty).replace_infer_with_error(interner);

        // If the deref chain contains a cycle (e.g. `A` derefs to `B` and `B` derefs to `A`), we
        // would revisit some already visited types. Stop here to avoid duplication.
        //
        // XXX: The recursion limit for `Autoderef` is currently 20, so `Vec::contains()` shouldn't
        // be too expensive. Replace this duplicate check with `FxHashSet` if it proves to be more
        // performant.
        if v.contains(&resolved) {
            break;
        }
        v.push(resolved);
    }
    v.into_iter()
}

pub(crate) trait TrackAutoderefSteps<'db>: Default + fmt::Debug {
    fn len(&self) -> usize;
    fn push(&mut self, ty: Ty<'db>, kind: AutoderefKind);
}

impl<'db> TrackAutoderefSteps<'db> for usize {
    fn len(&self) -> usize {
        *self
    }
    fn push(&mut self, _: Ty<'db>, _: AutoderefKind) {
        *self += 1;
    }
}
impl<'db> TrackAutoderefSteps<'db> for Vec<(Ty<'db>, AutoderefKind)> {
    fn len(&self) -> usize {
        self.len()
    }
    fn push(&mut self, ty: Ty<'db>, kind: AutoderefKind) {
        self.push((ty, kind));
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum AutoderefKind {
    /// A true pointer type, such as `&T` and `*mut T`.
    Builtin,
    /// A type which must dispatch to a `Deref` implementation.
    Overloaded,
}

struct AutoderefSnapshot<'db, Steps> {
    at_start: bool,
    reached_recursion_limit: bool,
    steps: Steps,
    cur_ty: Ty<'db>,
    obligations: PredicateObligations<'db>,
}

#[derive(Clone, Copy)]
struct AutoderefTraits {
    trait_: TraitId,
    trait_target: TypeAliasId,
}

// We use a trait here and a generic implementation unfortunately, because sometimes (specifically
// in place_op.rs), you need to have mutable access to the `InferenceContext` while the `Autoderef`
// borrows it.
pub(crate) trait AutoderefCtx<'db> {
    fn infcx(&self) -> &InferCtxt<'db>;
    fn param_env(&self) -> ParamEnv<'db>;
}

pub(crate) struct DefaultAutoderefCtx<'a, 'db> {
    infcx: &'a InferCtxt<'db>,
    param_env: ParamEnv<'db>,
}
impl<'db> AutoderefCtx<'db> for DefaultAutoderefCtx<'_, 'db> {
    #[inline]
    fn infcx(&self) -> &InferCtxt<'db> {
        self.infcx
    }
    #[inline]
    fn param_env(&self) -> ParamEnv<'db> {
        self.param_env
    }
}

pub(crate) struct InferenceContextAutoderefCtx<'a, 'b, 'db>(&'a mut InferenceContext<'b, 'db>);
impl<'db> AutoderefCtx<'db> for InferenceContextAutoderefCtx<'_, '_, 'db> {
    #[inline]
    fn infcx(&self) -> &InferCtxt<'db> {
        &self.0.table.infer_ctxt
    }
    #[inline]
    fn param_env(&self) -> ParamEnv<'db> {
        self.0.table.param_env
    }
}

/// Recursively dereference a type, considering both built-in
/// dereferences (`*`) and the `Deref` trait.
/// Although called `Autoderef` it can be configured to use the
/// `Receiver` trait instead of the `Deref` trait.
pub(crate) struct GeneralAutoderef<'db, Ctx, Steps = Vec<(Ty<'db>, AutoderefKind)>> {
    // Meta infos:
    ctx: Ctx,
    traits: Option<AutoderefTraits>,

    // Current state:
    state: AutoderefSnapshot<'db, Steps>,

    // Configurations:
    include_raw_pointers: bool,
    use_receiver_trait: bool,
}

pub(crate) type Autoderef<'a, 'db, Steps = Vec<(Ty<'db>, AutoderefKind)>> =
    GeneralAutoderef<'db, DefaultAutoderefCtx<'a, 'db>, Steps>;
pub(crate) type InferenceContextAutoderef<'a, 'b, 'db, Steps = Vec<(Ty<'db>, AutoderefKind)>> =
    GeneralAutoderef<'db, InferenceContextAutoderefCtx<'a, 'b, 'db>, Steps>;

impl<'db, Ctx, Steps> Iterator for GeneralAutoderef<'db, Ctx, Steps>
where
    Ctx: AutoderefCtx<'db>,
    Steps: TrackAutoderefSteps<'db>,
{
    type Item = (Ty<'db>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        debug!("autoderef: steps={:?}, cur_ty={:?}", self.state.steps, self.state.cur_ty);
        if self.state.at_start {
            self.state.at_start = false;
            debug!("autoderef stage #0 is {:?}", self.state.cur_ty);
            return Some((self.state.cur_ty, 0));
        }

        // If we have reached the recursion limit, error gracefully.
        if self.state.steps.len() >= AUTODEREF_RECURSION_LIMIT {
            self.state.reached_recursion_limit = true;
            return None;
        }

        if self.state.cur_ty.is_ty_var() {
            return None;
        }

        // Otherwise, deref if type is derefable:
        // NOTE: in the case of self.use_receiver_trait = true, you might think it would
        // be better to skip this clause and use the Overloaded case only, since &T
        // and &mut T implement Receiver. But built-in derefs apply equally to Receiver
        // and Deref, and this has benefits for const and the emitted MIR.
        let (kind, new_ty) =
            if let Some(ty) = self.state.cur_ty.builtin_deref(self.include_raw_pointers) {
                debug_assert_eq!(ty, self.infcx().resolve_vars_if_possible(ty));
                // NOTE: we may still need to normalize the built-in deref in case
                // we have some type like `&<Ty as Trait>::Assoc`, since users of
                // autoderef expect this type to have been structurally normalized.
                if let TyKind::Alias(..) = ty.kind() {
                    let (normalized_ty, obligations) =
                        structurally_normalize_ty(self.infcx(), self.param_env(), ty)?;
                    self.state.obligations.extend(obligations);
                    (AutoderefKind::Builtin, normalized_ty)
                } else {
                    (AutoderefKind::Builtin, ty)
                }
            } else if let Some(ty) = self.overloaded_deref_ty(self.state.cur_ty) {
                // The overloaded deref check already normalizes the pointee type.
                (AutoderefKind::Overloaded, ty)
            } else {
                return None;
            };

        self.state.steps.push(self.state.cur_ty, kind);
        debug!(
            "autoderef stage #{:?} is {:?} from {:?}",
            self.step_count(),
            new_ty,
            (self.state.cur_ty, kind)
        );
        self.state.cur_ty = new_ty;

        Some((self.state.cur_ty, self.step_count()))
    }
}

impl<'a, 'db> Autoderef<'a, 'db> {
    #[inline]
    pub(crate) fn new_with_tracking(
        infcx: &'a InferCtxt<'db>,
        param_env: ParamEnv<'db>,
        base_ty: Ty<'db>,
    ) -> Self {
        Self::new_impl(DefaultAutoderefCtx { infcx, param_env }, base_ty)
    }
}

impl<'a, 'b, 'db> InferenceContextAutoderef<'a, 'b, 'db> {
    #[inline]
    pub(crate) fn new_from_inference_context(
        ctx: &'a mut InferenceContext<'b, 'db>,
        base_ty: Ty<'db>,
    ) -> Self {
        Self::new_impl(InferenceContextAutoderefCtx(ctx), base_ty)
    }

    #[inline]
    pub(crate) fn ctx(&mut self) -> &mut InferenceContext<'b, 'db> {
        self.ctx.0
    }
}

impl<'a, 'db> Autoderef<'a, 'db, usize> {
    #[inline]
    pub(crate) fn new(
        infcx: &'a InferCtxt<'db>,
        param_env: ParamEnv<'db>,
        base_ty: Ty<'db>,
    ) -> Self {
        Self::new_impl(DefaultAutoderefCtx { infcx, param_env }, base_ty)
    }
}

impl<'db, Ctx, Steps> GeneralAutoderef<'db, Ctx, Steps>
where
    Ctx: AutoderefCtx<'db>,
    Steps: TrackAutoderefSteps<'db>,
{
    #[inline]
    fn new_impl(ctx: Ctx, base_ty: Ty<'db>) -> Self {
        GeneralAutoderef {
            state: AutoderefSnapshot {
                steps: Steps::default(),
                cur_ty: ctx.infcx().resolve_vars_if_possible(base_ty),
                obligations: PredicateObligations::new(),
                at_start: true,
                reached_recursion_limit: false,
            },
            ctx,
            traits: None,
            include_raw_pointers: false,
            use_receiver_trait: false,
        }
    }

    #[inline]
    fn infcx(&self) -> &InferCtxt<'db> {
        self.ctx.infcx()
    }

    #[inline]
    fn param_env(&self) -> ParamEnv<'db> {
        self.ctx.param_env()
    }

    #[inline]
    fn interner(&self) -> DbInterner<'db> {
        self.infcx().interner
    }

    fn autoderef_traits(&mut self) -> Option<AutoderefTraits> {
        let lang_items = self.interner().lang_items();
        match &mut self.traits {
            Some(it) => Some(*it),
            None => {
                let traits = if self.use_receiver_trait {
                    (|| {
                        Some(AutoderefTraits {
                            trait_: lang_items.Receiver?,
                            trait_target: lang_items.ReceiverTarget?,
                        })
                    })()
                    .or_else(|| {
                        Some(AutoderefTraits {
                            trait_: lang_items.Deref?,
                            trait_target: lang_items.DerefTarget?,
                        })
                    })?
                } else {
                    AutoderefTraits {
                        trait_: lang_items.Deref?,
                        trait_target: lang_items.DerefTarget?,
                    }
                };
                Some(*self.traits.insert(traits))
            }
        }
    }

    fn overloaded_deref_ty(&mut self, ty: Ty<'db>) -> Option<Ty<'db>> {
        debug!("overloaded_deref_ty({:?})", ty);
        let interner = self.interner();

        // <ty as Deref>, or whatever the equivalent trait is that we've been asked to walk.
        let AutoderefTraits { trait_, trait_target } = self.autoderef_traits()?;

        let trait_ref = TraitRef::new(interner, trait_.into(), [ty]);
        let obligation =
            Obligation::new(interner, ObligationCause::new(), self.param_env(), trait_ref);
        // We detect whether the self type implements `Deref` before trying to
        // structurally normalize. We use `predicate_may_hold_opaque_types_jank`
        // to support not-yet-defined opaque types. It will succeed for `impl Deref`
        // but fail for `impl OtherTrait`.
        if !self.infcx().predicate_may_hold_opaque_types_jank(&obligation) {
            debug!("overloaded_deref_ty: cannot match obligation");
            return None;
        }

        let (normalized_ty, obligations) = structurally_normalize_ty(
            self.infcx(),
            self.param_env(),
            Ty::new_projection(interner, trait_target.into(), [ty]),
        )?;
        debug!("overloaded_deref_ty({:?}) = ({:?}, {:?})", ty, normalized_ty, obligations);
        self.state.obligations.extend(obligations);

        Some(self.infcx().resolve_vars_if_possible(normalized_ty))
    }

    /// Returns the final type we ended up with, which may be an unresolved
    /// inference variable.
    pub(crate) fn final_ty(&self) -> Ty<'db> {
        self.state.cur_ty
    }

    pub(crate) fn step_count(&self) -> usize {
        self.state.steps.len()
    }

    pub(crate) fn take_obligations(&mut self) -> PredicateObligations<'db> {
        std::mem::take(&mut self.state.obligations)
    }

    pub(crate) fn steps(&self) -> &Steps {
        &self.state.steps
    }

    pub(crate) fn reached_recursion_limit(&self) -> bool {
        self.state.reached_recursion_limit
    }

    /// also dereference through raw pointer types
    /// e.g., assuming ptr_to_Foo is the type `*const Foo`
    /// fcx.autoderef(span, ptr_to_Foo)  => [*const Foo]
    /// fcx.autoderef(span, ptr_to_Foo).include_raw_ptrs() => [*const Foo, Foo]
    pub(crate) fn include_raw_pointers(mut self) -> Self {
        self.include_raw_pointers = true;
        self
    }

    /// Use `core::ops::Receiver` and `core::ops::Receiver::Target` as
    /// the trait and associated type to iterate, instead of
    /// `core::ops::Deref` and `core::ops::Deref::Target`
    pub(crate) fn use_receiver_trait(mut self) -> Self {
        self.use_receiver_trait = true;
        self
    }
}

fn structurally_normalize_ty<'db>(
    infcx: &InferCtxt<'db>,
    param_env: ParamEnv<'db>,
    ty: Ty<'db>,
) -> Option<(Ty<'db>, PredicateObligations<'db>)> {
    let mut ocx = ObligationCtxt::new(infcx);
    let Ok(normalized_ty) = ocx.structurally_normalize_ty(&ObligationCause::misc(), param_env, ty)
    else {
        // We shouldn't have errors here in the old solver, except for
        // evaluate/fulfill mismatches, but that's not a reason for an ICE.
        return None;
    };
    let errors = ocx.try_evaluate_obligations();
    if !errors.is_empty() {
        unreachable!();
    }

    Some((normalized_ty, ocx.into_pending_obligations()))
}
