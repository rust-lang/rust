use crate::traits::query::evaluate_obligation::InferCtxtExt;
use crate::traits::{self, TraitEngine};
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_infer::infer::InferCtxt;
use rustc_middle::ty::{self, TraitRef, Ty, TyCtxt, WithConstness};
use rustc_middle::ty::{ToPredicate, TypeFoldable};
use rustc_session::DiagnosticMessageId;
use rustc_span::def_id::LOCAL_CRATE;
use rustc_span::Span;

#[derive(Copy, Clone, Debug)]
pub enum AutoderefKind {
    Builtin,
    Overloaded,
}

struct AutoderefSnapshot<'tcx> {
    at_start: bool,
    reached_recursion_limit: bool,
    steps: Vec<(Ty<'tcx>, AutoderefKind)>,
    cur_ty: Ty<'tcx>,
    obligations: Vec<traits::PredicateObligation<'tcx>>,
}

pub struct Autoderef<'a, 'tcx> {
    // Meta infos:
    infcx: &'a InferCtxt<'a, 'tcx>,
    span: Span,
    overloaded_span: Span,
    body_id: hir::HirId,
    param_env: ty::ParamEnv<'tcx>,

    // Current state:
    state: AutoderefSnapshot<'tcx>,

    // Configurations:
    include_raw_pointers: bool,
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
        let (kind, new_ty) =
            if let Some(mt) = self.state.cur_ty.builtin_deref(self.include_raw_pointers) {
                (AutoderefKind::Builtin, mt.ty)
            } else if let Some(ty) = self.overloaded_deref_ty(self.state.cur_ty) {
                (AutoderefKind::Overloaded, ty)
            } else {
                return None;
            };

        if new_ty.references_error() {
            return None;
        }

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
        infcx: &'a InferCtxt<'a, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        body_id: hir::HirId,
        span: Span,
        base_ty: Ty<'tcx>,
        overloaded_span: Span,
    ) -> Autoderef<'a, 'tcx> {
        Autoderef {
            infcx,
            span,
            overloaded_span,
            body_id,
            param_env,
            state: AutoderefSnapshot {
                steps: vec![],
                cur_ty: infcx.resolve_vars_if_possible(base_ty),
                obligations: vec![],
                at_start: true,
                reached_recursion_limit: false,
            },
            include_raw_pointers: false,
            silence_errors: false,
        }
    }

    fn overloaded_deref_ty(&mut self, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
        debug!("overloaded_deref_ty({:?})", ty);

        let tcx = self.infcx.tcx;

        // <ty as Deref>
        let trait_ref = TraitRef {
            def_id: tcx.lang_items().deref_trait()?,
            substs: tcx.mk_substs_trait(ty, &[]),
        };

        let cause = traits::ObligationCause::misc(self.span, self.body_id);

        let obligation = traits::Obligation::new(
            cause.clone(),
            self.param_env,
            ty::Binder::dummy(trait_ref).without_const().to_predicate(tcx),
        );
        if !self.infcx.predicate_may_hold(&obligation) {
            debug!("overloaded_deref_ty: cannot match obligation");
            return None;
        }

        let mut fulfillcx = traits::FulfillmentContext::new_in_snapshot();
        let normalized_ty = fulfillcx.normalize_projection_type(
            &self.infcx,
            self.param_env,
            ty::ProjectionTy {
                item_def_id: tcx.lang_items().deref_target()?,
                substs: trait_ref.substs,
            },
            cause,
        );
        if let Err(e) = fulfillcx.select_where_possible(&self.infcx) {
            // This shouldn't happen, except for evaluate/fulfill mismatches,
            // but that's not a reason for an ICE (`predicate_may_hold` is conservative
            // by design).
            debug!("overloaded_deref_ty: encountered errors {:?} while fulfilling", e);
            return None;
        }
        let obligations = fulfillcx.pending_obligations();
        debug!("overloaded_deref_ty({:?}) = ({:?}, {:?})", ty, normalized_ty, obligations);
        self.state.obligations.extend(obligations);

        Some(self.infcx.resolve_vars_if_possible(normalized_ty))
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

    pub fn into_obligations(self) -> Vec<traits::PredicateObligation<'tcx>> {
        self.state.obligations
    }

    pub fn steps(&self) -> &[(Ty<'tcx>, AutoderefKind)] {
        &self.state.steps
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn overloaded_span(&self) -> Span {
        self.overloaded_span
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

    pub fn silence_errors(mut self) -> Self {
        self.silence_errors = true;
        self
    }
}

pub fn report_autoderef_recursion_limit_error<'tcx>(tcx: TyCtxt<'tcx>, span: Span, ty: Ty<'tcx>) {
    // We've reached the recursion limit, error gracefully.
    let suggested_limit = tcx.recursion_limit() * 2;
    let msg = format!("reached the recursion limit while auto-dereferencing `{:?}`", ty);
    let error_id = (DiagnosticMessageId::ErrorId(55), Some(span), msg);
    let fresh = tcx.sess.one_time_diagnostics.borrow_mut().insert(error_id);
    if fresh {
        struct_span_err!(
            tcx.sess,
            span,
            E0055,
            "reached the recursion limit while auto-dereferencing `{:?}`",
            ty
        )
        .span_label(span, "deref recursion limit reached")
        .help(&format!(
            "consider adding a `#![recursion_limit=\"{}\"]` attribute to your crate (`{}`)",
            suggested_limit,
            tcx.crate_name(LOCAL_CRATE),
        ))
        .emit();
    }
}
