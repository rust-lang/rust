//! In certain situations, rust automatically inserts derefs as necessary: for
//! example, field accesses `foo.bar` still work when `foo` is actually a
//! reference to a type with the field `bar`. This is an approximation of the
//! logic in rustc (which lives in rustc_hir_analysis/check/autoderef.rs).

use std::fmt;

use hir_def::{TraitId, TypeAliasId, lang_item::LangItem};
use rustc_type_ir::inherent::{IntoKind, Ty as _};
use tracing::debug;
use triomphe::Arc;

use crate::{
    TraitEnvironment,
    db::HirDatabase,
    infer::unify::InferenceTable,
    next_solver::{
        Canonical, TraitRef, Ty, TyKind,
        infer::{
            InferOk,
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
    env: Arc<TraitEnvironment<'db>>,
    ty: Canonical<'db, Ty<'db>>,
) -> impl Iterator<Item = Ty<'db>> + use<'db> {
    let mut table = InferenceTable::new(db, env, None);
    let ty = table.instantiate_canonical(ty);
    let mut autoderef = Autoderef::new_no_tracking(&mut table, ty);
    let mut v = Vec::new();
    while let Some((ty, _steps)) = autoderef.next() {
        // `ty` may contain unresolved inference variables. Since there's no chance they would be
        // resolved, just replace with fallback type.
        let resolved = autoderef.table.resolve_completely(ty);

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

/// Recursively dereference a type, considering both built-in
/// dereferences (`*`) and the `Deref` trait.
/// Although called `Autoderef` it can be configured to use the
/// `Receiver` trait instead of the `Deref` trait.
pub(crate) struct Autoderef<'a, 'db, Steps = Vec<(Ty<'db>, AutoderefKind)>> {
    // Meta infos:
    pub(crate) table: &'a mut InferenceTable<'db>,
    traits: Option<AutoderefTraits>,

    // Current state:
    state: AutoderefSnapshot<'db, Steps>,

    // Configurations:
    include_raw_pointers: bool,
    use_receiver_trait: bool,
}

impl<'a, 'db, Steps: TrackAutoderefSteps<'db>> Iterator for Autoderef<'a, 'db, Steps> {
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
        let (kind, new_ty) = if let Some(ty) =
            self.state.cur_ty.builtin_deref(self.table.db, self.include_raw_pointers)
        {
            debug_assert_eq!(ty, self.table.infer_ctxt.resolve_vars_if_possible(ty));
            // NOTE: we may still need to normalize the built-in deref in case
            // we have some type like `&<Ty as Trait>::Assoc`, since users of
            // autoderef expect this type to have been structurally normalized.
            if let TyKind::Alias(..) = ty.kind() {
                let (normalized_ty, obligations) = structurally_normalize_ty(self.table, ty)?;
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
    pub(crate) fn new(table: &'a mut InferenceTable<'db>, base_ty: Ty<'db>) -> Self {
        Self::new_impl(table, base_ty)
    }
}

impl<'a, 'db> Autoderef<'a, 'db, usize> {
    pub(crate) fn new_no_tracking(table: &'a mut InferenceTable<'db>, base_ty: Ty<'db>) -> Self {
        Self::new_impl(table, base_ty)
    }
}

impl<'a, 'db, Steps: TrackAutoderefSteps<'db>> Autoderef<'a, 'db, Steps> {
    fn new_impl(table: &'a mut InferenceTable<'db>, base_ty: Ty<'db>) -> Self {
        Autoderef {
            state: AutoderefSnapshot {
                steps: Steps::default(),
                cur_ty: table.infer_ctxt.resolve_vars_if_possible(base_ty),
                obligations: PredicateObligations::new(),
                at_start: true,
                reached_recursion_limit: false,
            },
            table,
            traits: None,
            include_raw_pointers: false,
            use_receiver_trait: false,
        }
    }

    fn autoderef_traits(&mut self) -> Option<AutoderefTraits> {
        match &mut self.traits {
            Some(it) => Some(*it),
            None => {
                let traits = if self.use_receiver_trait {
                    (|| {
                        Some(AutoderefTraits {
                            trait_: LangItem::Receiver
                                .resolve_trait(self.table.db, self.table.trait_env.krate)?,
                            trait_target: LangItem::ReceiverTarget
                                .resolve_type_alias(self.table.db, self.table.trait_env.krate)?,
                        })
                    })()
                    .or_else(|| {
                        Some(AutoderefTraits {
                            trait_: LangItem::Deref
                                .resolve_trait(self.table.db, self.table.trait_env.krate)?,
                            trait_target: LangItem::DerefTarget
                                .resolve_type_alias(self.table.db, self.table.trait_env.krate)?,
                        })
                    })?
                } else {
                    AutoderefTraits {
                        trait_: LangItem::Deref
                            .resolve_trait(self.table.db, self.table.trait_env.krate)?,
                        trait_target: LangItem::DerefTarget
                            .resolve_type_alias(self.table.db, self.table.trait_env.krate)?,
                    }
                };
                Some(*self.traits.insert(traits))
            }
        }
    }

    fn overloaded_deref_ty(&mut self, ty: Ty<'db>) -> Option<Ty<'db>> {
        debug!("overloaded_deref_ty({:?})", ty);
        let interner = self.table.interner();

        // <ty as Deref>, or whatever the equivalent trait is that we've been asked to walk.
        let AutoderefTraits { trait_, trait_target } = self.autoderef_traits()?;

        let trait_ref = TraitRef::new(interner, trait_.into(), [ty]);
        let obligation =
            Obligation::new(interner, ObligationCause::new(), self.table.trait_env.env, trait_ref);
        // We detect whether the self type implements `Deref` before trying to
        // structurally normalize. We use `predicate_may_hold_opaque_types_jank`
        // to support not-yet-defined opaque types. It will succeed for `impl Deref`
        // but fail for `impl OtherTrait`.
        if !self.table.infer_ctxt.predicate_may_hold_opaque_types_jank(&obligation) {
            debug!("overloaded_deref_ty: cannot match obligation");
            return None;
        }

        let (normalized_ty, obligations) = structurally_normalize_ty(
            self.table,
            Ty::new_projection(interner, trait_target.into(), [ty]),
        )?;
        debug!("overloaded_deref_ty({:?}) = ({:?}, {:?})", ty, normalized_ty, obligations);
        self.state.obligations.extend(obligations);

        Some(self.table.infer_ctxt.resolve_vars_if_possible(normalized_ty))
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

    #[expect(dead_code)]
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
    table: &InferenceTable<'db>,
    ty: Ty<'db>,
) -> Option<(Ty<'db>, PredicateObligations<'db>)> {
    let mut ocx = ObligationCtxt::new(&table.infer_ctxt);
    let Ok(normalized_ty) =
        ocx.structurally_normalize_ty(&ObligationCause::misc(), table.trait_env.env, ty)
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

pub(crate) fn overloaded_deref_ty<'db>(
    table: &InferenceTable<'db>,
    ty: Ty<'db>,
) -> Option<InferOk<'db, Ty<'db>>> {
    let interner = table.interner();

    let trait_target = LangItem::DerefTarget.resolve_type_alias(table.db, table.trait_env.krate)?;

    let (normalized_ty, obligations) =
        structurally_normalize_ty(table, Ty::new_projection(interner, trait_target.into(), [ty]))?;

    Some(InferOk { value: normalized_ty, obligations })
}
