use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::PredicateObligations;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt};
use rustc_session::Limit;
use rustc_span::def_id::{LOCAL_CRATE, LocalDefId};
use rustc_span::{ErrorGuaranteed, Span};
use rustc_trait_selection::traits::ObligationCtxt;
use tracing::{debug, instrument};

use crate::errors::AutoDerefReachedRecursionLimit;
use crate::traits;
use crate::traits::query::evaluate_obligation::InferCtxtExt;

#[derive(Copy, Clone, Debug)]
pub enum AutoderefKind {
    /// A true pointer type, such as `&T` and `*mut T`.
    Builtin,
    /// A type which must dispatch to a `Deref` implementation.
    Overloaded,
}
struct AutoderefSnapshot<'tcx> {
    at_start: bool,
    reached_recursion_limit: bool,
    steps: Vec<(Ty<'tcx>, AutoderefKind)>,
    cur_ty: Ty<'tcx>,
    obligations: PredicateObligations<'tcx>,
}

/// Recursively dereference a type, considering both built-in
/// dereferences (`*`) and the `Deref` trait.
/// Although called `Autoderef` it can be configured to use the
/// `Receiver` trait instead of the `Deref` trait.
pub struct Autoderef<'a, 'tcx> {
    // Meta infos:
    infcx: &'a InferCtxt<'tcx>,
    span: Span,
    body_id: LocalDefId,
    param_env: ty::ParamEnv<'tcx>,

    // Current state:
    state: AutoderefSnapshot<'tcx>,

    // Configurations:
    include_raw_pointers: bool,
    use_receiver_trait: bool,
    silence_errors: bool,
}

impl<'a, 'tcx> Iterator for Autoderef<'a, 'tcx> {
    type Item = (Ty<'tcx>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let tcx = self.infcx.tcx;

        debug!("autoderef: steps={:?}, cur_ty={:?}", self.state.steps, self.state.cur_ty);
        if self.state.at_start {
            self.state.at_start = false;
            debug!("autoderef stage #0 is {:?}", self.state.cur_ty);
            return Some((self.state.cur_ty, 0));
        }

        // If we have reached the recursion limit, error gracefully.
        if !tcx.recursion_limit().value_within_limit(self.state.steps.len()) {
            if !self.silence_errors {
                report_autoderef_recursion_limit_error(tcx, self.span, self.state.cur_ty);
            }
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
                debug_assert_eq!(ty, self.infcx.resolve_vars_if_possible(ty));
                // NOTE: we may still need to normalize the built-in deref in case
                // we have some type like `&<Ty as Trait>::Assoc`, since users of
                // autoderef expect this type to have been structurally normalized.
                if self.infcx.next_trait_solver()
                    && let ty::Alias(..) = ty.kind()
                {
                    let (normalized_ty, obligations) = self.structurally_normalize_ty(ty)?;
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

        self.state.steps.push((self.state.cur_ty, kind));
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

impl<'a, 'tcx> Autoderef<'a, 'tcx> {
    pub fn new(
        infcx: &'a InferCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        body_def_id: LocalDefId,
        span: Span,
        base_ty: Ty<'tcx>,
    ) -> Self {
        Autoderef {
            infcx,
            span,
            body_id: body_def_id,
            param_env,
            state: AutoderefSnapshot {
                steps: vec![],
                cur_ty: infcx.resolve_vars_if_possible(base_ty),
                obligations: PredicateObligations::new(),
                at_start: true,
                reached_recursion_limit: false,
            },
            include_raw_pointers: false,
            use_receiver_trait: false,
            silence_errors: false,
        }
    }

    fn overloaded_deref_ty(&mut self, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
        debug!("overloaded_deref_ty({:?})", ty);
        let tcx = self.infcx.tcx;

        if ty.references_error() {
            return None;
        }

        // <ty as Deref>, or whatever the equivalent trait is that we've been asked to walk.
        let (trait_def_id, trait_target_def_id) = if self.use_receiver_trait {
            (tcx.lang_items().receiver_trait()?, tcx.lang_items().receiver_target()?)
        } else {
            (tcx.lang_items().deref_trait()?, tcx.lang_items().deref_target()?)
        };
        let trait_ref = ty::TraitRef::new(tcx, trait_def_id, [ty]);
        let cause = traits::ObligationCause::misc(self.span, self.body_id);
        let obligation = traits::Obligation::new(
            tcx,
            cause.clone(),
            self.param_env,
            ty::Binder::dummy(trait_ref),
        );
        if !self.infcx.predicate_may_hold(&obligation) {
            debug!("overloaded_deref_ty: cannot match obligation");
            return None;
        }

        let (normalized_ty, obligations) =
            self.structurally_normalize_ty(Ty::new_projection(tcx, trait_target_def_id, [ty]))?;
        debug!("overloaded_deref_ty({:?}) = ({:?}, {:?})", ty, normalized_ty, obligations);
        self.state.obligations.extend(obligations);

        Some(self.infcx.resolve_vars_if_possible(normalized_ty))
    }

    #[instrument(level = "debug", skip(self), ret)]
    pub fn structurally_normalize_ty(
        &self,
        ty: Ty<'tcx>,
    ) -> Option<(Ty<'tcx>, PredicateObligations<'tcx>)> {
        let ocx = ObligationCtxt::new(self.infcx);
        let Ok(normalized_ty) = ocx.structurally_normalize_ty(
            &traits::ObligationCause::misc(self.span, self.body_id),
            self.param_env,
            ty,
        ) else {
            // We shouldn't have errors here, except for evaluate/fulfill mismatches,
            // but that's not a reason for an ICE (`predicate_may_hold` is conservative
            // by design).
            // FIXME(-Znext-solver): This *actually* shouldn't happen then.
            return None;
        };
        let errors = ocx.select_where_possible();
        if !errors.is_empty() {
            // This shouldn't happen, except for evaluate/fulfill mismatches,
            // but that's not a reason for an ICE (`predicate_may_hold` is conservative
            // by design).
            debug!(?errors, "encountered errors while fulfilling");
            return None;
        }

        Some((normalized_ty, ocx.into_pending_obligations()))
    }

    /// Returns the final type we ended up with, which may be an inference
    /// variable (we will resolve it first, if we want).
    pub fn final_ty(&self, resolve: bool) -> Ty<'tcx> {
        if resolve {
            self.infcx.resolve_vars_if_possible(self.state.cur_ty)
        } else {
            self.state.cur_ty
        }
    }

    pub fn step_count(&self) -> usize {
        self.state.steps.len()
    }

    pub fn into_obligations(self) -> PredicateObligations<'tcx> {
        self.state.obligations
    }

    pub fn current_obligations(&self) -> PredicateObligations<'tcx> {
        self.state.obligations.clone()
    }

    pub fn steps(&self) -> &[(Ty<'tcx>, AutoderefKind)] {
        &self.state.steps
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn reached_recursion_limit(&self) -> bool {
        self.state.reached_recursion_limit
    }

    /// also dereference through raw pointer types
    /// e.g., assuming ptr_to_Foo is the type `*const Foo`
    /// fcx.autoderef(span, ptr_to_Foo)  => [*const Foo]
    /// fcx.autoderef(span, ptr_to_Foo).include_raw_ptrs() => [*const Foo, Foo]
    pub fn include_raw_pointers(mut self) -> Self {
        self.include_raw_pointers = true;
        self
    }

    /// Use `core::ops::Receiver` and `core::ops::Receiver::Target` as
    /// the trait and associated type to iterate, instead of
    /// `core::ops::Deref` and `core::ops::Deref::Target`
    pub fn use_receiver_trait(mut self) -> Self {
        self.use_receiver_trait = true;
        self
    }

    pub fn silence_errors(mut self) -> Self {
        self.silence_errors = true;
        self
    }
}

pub fn report_autoderef_recursion_limit_error<'tcx>(
    tcx: TyCtxt<'tcx>,
    span: Span,
    ty: Ty<'tcx>,
) -> ErrorGuaranteed {
    // We've reached the recursion limit, error gracefully.
    let suggested_limit = match tcx.recursion_limit() {
        Limit(0) => Limit(2),
        limit => limit * 2,
    };
    tcx.dcx().emit_err(AutoDerefReachedRecursionLimit {
        span,
        ty,
        suggested_limit,
        crate_name: tcx.crate_name(LOCAL_CRATE),
    })
}
