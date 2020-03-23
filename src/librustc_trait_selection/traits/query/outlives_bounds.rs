use crate::infer::canonical::OriginalQueryValues;
use crate::infer::InferCtxt;
use crate::traits::query::NoSolution;
use crate::traits::{FulfillmentContext, ObligationCause, TraitEngine};
use rustc::ty::{self, Ty};
use rustc_hir as hir;
use rustc_infer::traits::TraitEngineExt as _;
use rustc_span::source_map::Span;

pub use rustc::traits::query::OutlivesBound;

pub trait InferCtxtExt<'tcx> {
    fn implied_outlives_bounds(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        body_id: hir::HirId,
        ty: Ty<'tcx>,
        span: Span,
    ) -> Vec<OutlivesBound<'tcx>>;
}

impl<'cx, 'tcx> InferCtxtExt<'tcx> for InferCtxt<'cx, 'tcx> {
    /// Implied bounds are region relationships that we deduce
    /// automatically. The idea is that (e.g.) a caller must check that a
    /// function's argument types are well-formed immediately before
    /// calling that fn, and hence the *callee* can assume that its
    /// argument types are well-formed. This may imply certain relationships
    /// between generic parameters. For example:
    ///
    ///     fn foo<'a,T>(x: &'a T)
    ///
    /// can only be called with a `'a` and `T` such that `&'a T` is WF.
    /// For `&'a T` to be WF, `T: 'a` must hold. So we can assume `T: 'a`.
    ///
    /// # Parameters
    ///
    /// - `param_env`, the where-clauses in scope
    /// - `body_id`, the body-id to use when normalizing assoc types.
    ///   Note that this may cause outlives obligations to be injected
    ///   into the inference context with this body-id.
    /// - `ty`, the type that we are supposed to assume is WF.
    /// - `span`, a span to use when normalizing, hopefully not important,
    ///   might be useful if a `bug!` occurs.
    fn implied_outlives_bounds(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        body_id: hir::HirId,
        ty: Ty<'tcx>,
        span: Span,
    ) -> Vec<OutlivesBound<'tcx>> {
        debug!("implied_outlives_bounds(ty = {:?})", ty);

        let mut orig_values = OriginalQueryValues::default();
        let key = self.canonicalize_query(&param_env.and(ty), &mut orig_values);
        let result = match self.tcx.implied_outlives_bounds(key) {
            Ok(r) => r,
            Err(NoSolution) => {
                self.tcx.sess.delay_span_bug(
                    span,
                    "implied_outlives_bounds failed to solve all obligations",
                );
                return vec![];
            }
        };
        assert!(result.value.is_proven());

        let result = self.instantiate_query_response_and_region_obligations(
            &ObligationCause::misc(span, body_id),
            param_env,
            &orig_values,
            &result,
        );
        debug!("implied_outlives_bounds for {:?}: {:#?}", ty, result);
        let result = match result {
            Ok(v) => v,
            Err(_) => {
                self.tcx.sess.delay_span_bug(span, "implied_outlives_bounds failed to instantiate");
                return vec![];
            }
        };

        // Instantiation may have produced new inference variables and constraints on those
        // variables. Process these constraints.
        let mut fulfill_cx = FulfillmentContext::new();
        fulfill_cx.register_predicate_obligations(self, result.obligations);
        if fulfill_cx.select_all_or_error(self).is_err() {
            self.tcx.sess.delay_span_bug(
                span,
                "implied_outlives_bounds failed to solve obligations from instantiation",
            );
        }

        result.value
    }
}
