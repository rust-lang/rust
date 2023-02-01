use crate::infer::InferCtxt;
use crate::traits::query::type_op::{self, TypeOp, TypeOpOutput};
use crate::traits::query::NoSolution;
use crate::traits::{ObligationCause, ObligationCtxt};
use rustc_data_structures::fx::FxIndexSet;
use rustc_infer::infer::resolve::OpportunisticRegionResolver;
use rustc_middle::ty::{self, ParamEnv, Ty, TypeFolder, TypeVisitableExt};
use rustc_span::def_id::LocalDefId;

pub use rustc_middle::traits::query::OutlivesBound;

type Bounds<'a, 'tcx: 'a> = impl Iterator<Item = OutlivesBound<'tcx>> + 'a;
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
        assert!(!ty.needs_infer());

        let span = self.tcx.def_span(body_id);
        let result = param_env
            .and(type_op::implied_outlives_bounds::ImpliedOutlivesBounds { ty })
            .fully_perform(self);
        let result = match result {
            Ok(r) => r,
            Err(NoSolution) => {
                self.tcx.sess.delay_span_bug(
                    span,
                    "implied_outlives_bounds failed to solve all obligations",
                );
                return vec![];
            }
        };

        let TypeOpOutput { output, constraints, .. } = result;

        if let Some(constraints) = constraints {
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

        output
    }

    fn implied_bounds_tys(
        &'a self,
        param_env: ParamEnv<'tcx>,
        body_id: LocalDefId,
        tys: FxIndexSet<Ty<'tcx>>,
    ) -> Bounds<'a, 'tcx> {
        tys.into_iter()
            .map(move |ty| self.implied_outlives_bounds(param_env, body_id, ty))
            .flatten()
    }
}
