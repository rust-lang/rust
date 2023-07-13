use crate::infer::InferCtxt;
use crate::traits::{ObligationCause, ObligationCtxt};
use rustc_data_structures::fx::FxIndexSet;
use rustc_infer::infer::resolve::OpportunisticRegionResolver;
use rustc_infer::infer::InferOk;
use rustc_middle::infer::canonical::{OriginalQueryValues, QueryRegionConstraints};
use rustc_middle::ty::{self, ParamEnv, Ty, TypeFolder, TypeVisitableExt};
use rustc_span::def_id::LocalDefId;

pub use rustc_middle::traits::query::OutlivesBound;

pub type Bounds<'a, 'tcx: 'a> = impl Iterator<Item = OutlivesBound<'tcx>> + 'a;
pub trait InferCtxtExt<'a, 'tcx> {
    fn implied_outlives_bounds(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        body_id: LocalDefId,
        ty: Ty<'tcx>,
    ) -> Vec<OutlivesBound<'tcx>>;

    fn implied_bounds_tys(
        &'a self,
        param_env: ty::ParamEnv<'tcx>,
        body_id: LocalDefId,
        tys: FxIndexSet<Ty<'tcx>>,
    ) -> Bounds<'a, 'tcx>;
}

impl<'a, 'tcx: 'a> InferCtxtExt<'a, 'tcx> for InferCtxt<'tcx> {
    /// Implied bounds are region relationships that we deduce
    /// automatically. The idea is that (e.g.) a caller must check that a
    /// function's argument types are well-formed immediately before
    /// calling that fn, and hence the *callee* can assume that its
    /// argument types are well-formed. This may imply certain relationships
    /// between generic parameters. For example:
    /// ```
    /// fn foo<T>(x: &T) {}
    /// ```
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
    #[instrument(level = "debug", skip(self, param_env, body_id), ret)]
    fn implied_outlives_bounds(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        body_id: LocalDefId,
        ty: Ty<'tcx>,
    ) -> Vec<OutlivesBound<'tcx>> {
        let ty = self.resolve_vars_if_possible(ty);
        let ty = OpportunisticRegionResolver::new(self).fold_ty(ty);

        // We do not expect existential variables in implied bounds.
        // We may however encounter unconstrained lifetime variables in invalid
        // code. See #110161 for context.
        assert!(!ty.has_non_region_infer());
        if ty.has_infer() {
            self.tcx.sess.delay_span_bug(
                self.tcx.def_span(body_id),
                "skipped implied_outlives_bounds due to unconstrained lifetimes",
            );
            return vec![];
        }

        let mut canonical_var_values = OriginalQueryValues::default();
        let canonical_ty =
            self.canonicalize_query_keep_static(param_env.and(ty), &mut canonical_var_values);
        let Ok(canonical_result) = self.tcx.implied_outlives_bounds(canonical_ty) else {
            return vec![];
        };

        let mut constraints = QueryRegionConstraints::default();
        let Ok(InferOk { value, obligations }) = self
            .instantiate_nll_query_response_and_region_obligations(
                &ObligationCause::dummy(),
                param_env,
                &canonical_var_values,
                canonical_result,
                &mut constraints,
            )
        else {
            return vec![];
        };
        assert_eq!(&obligations, &[]);

        if !constraints.is_empty() {
            let span = self.tcx.def_span(body_id);

            debug!(?constraints);
            if !constraints.member_constraints.is_empty() {
                span_bug!(span, "{:#?}", constraints.member_constraints);
            }

            // Instantiation may have produced new inference variables and constraints on those
            // variables. Process these constraints.
            let ocx = ObligationCtxt::new(self);
            let cause = ObligationCause::misc(span, body_id);
            for &constraint in &constraints.outlives {
                ocx.register_obligation(self.query_outlives_constraint_to_obligation(
                    constraint,
                    cause.clone(),
                    param_env,
                ));
            }

            let errors = ocx.select_all_or_error();
            if !errors.is_empty() {
                self.tcx.sess.delay_span_bug(
                    span,
                    "implied_outlives_bounds failed to solve obligations from instantiation",
                );
            }
        };

        value
    }

    fn implied_bounds_tys(
        &'a self,
        param_env: ParamEnv<'tcx>,
        body_id: LocalDefId,
        tys: FxIndexSet<Ty<'tcx>>,
    ) -> Bounds<'a, 'tcx> {
        tys.into_iter().flat_map(move |ty| self.implied_outlives_bounds(param_env, body_id, ty))
    }
}
