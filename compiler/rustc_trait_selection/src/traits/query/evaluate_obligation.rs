use rustc_infer::traits::solve::Goal;
use rustc_macros::extension;
use rustc_middle::{span_bug, ty};
use rustc_next_trait_solver::solve::SolverDelegateEvalExt;

use crate::infer::InferCtxt;
use crate::infer::canonical::OriginalQueryValues;
use crate::solve::SolverDelegate;
use crate::traits::{
    EvaluationResult, ObligationCtxt, OverflowError, PredicateObligation, SelectionContext,
};

#[extension(pub trait InferCtxtExt<'tcx>)]
impl<'tcx> InferCtxt<'tcx> {
    /// Evaluates whether the predicate can be satisfied (by any means)
    /// in the given `ParamEnv`.
    fn predicate_may_hold(&self, obligation: &PredicateObligation<'tcx>) -> bool {
        self.evaluate_obligation_no_overflow(obligation).may_apply()
    }

    /// See the comment on [OpaqueTypesJank](crate::solve::OpaqueTypesJank)
    /// for more details.
    fn predicate_may_hold_opaque_types_jank(&self, obligation: &PredicateObligation<'tcx>) -> bool {
        if self.next_trait_solver() {
            self.goal_may_hold_opaque_types_jank(Goal::new(
                self.tcx,
                obligation.param_env,
                obligation.predicate,
            ))
        } else {
            self.predicate_may_hold(obligation)
        }
    }

    /// See the comment on [OpaqueTypesJank](crate::solve::OpaqueTypesJank)
    /// for more details.
    fn goal_may_hold_opaque_types_jank(&self, goal: Goal<'tcx, ty::Predicate<'tcx>>) -> bool {
        assert!(self.next_trait_solver());
        <&SolverDelegate<'tcx>>::from(self).root_goal_may_hold_opaque_types_jank(goal)
    }

    /// Evaluates whether the predicate can be satisfied in the given
    /// `ParamEnv`, and returns `false` if not certain. However, this is
    /// not entirely accurate if inference variables are involved.
    ///
    /// This version may conservatively fail when outlives obligations
    /// are required. Therefore, this version should only be used for
    /// optimizations or diagnostics and be treated as if it can always
    /// return `false`.
    ///
    /// # Example
    ///
    /// ```
    /// # #![allow(dead_code)]
    /// trait Trait {}
    ///
    /// fn check<T: Trait>() {}
    ///
    /// fn foo<T: 'static>()
    /// where
    ///     &'static T: Trait,
    /// {
    ///     // Evaluating `&'?0 T: Trait` adds a `'?0: 'static` outlives obligation,
    ///     // which means that `predicate_must_hold_considering_regions` will return
    ///     // `false`.
    ///     check::<&'_ T>();
    /// }
    /// ```
    fn predicate_must_hold_considering_regions(
        &self,
        obligation: &PredicateObligation<'tcx>,
    ) -> bool {
        self.evaluate_obligation_no_overflow(obligation).must_apply_considering_regions()
    }

    /// Evaluates whether the predicate can be satisfied in the given
    /// `ParamEnv`, and returns `false` if not certain. However, this is
    /// not entirely accurate if inference variables are involved.
    ///
    /// This version ignores all outlives constraints.
    fn predicate_must_hold_modulo_regions(&self, obligation: &PredicateObligation<'tcx>) -> bool {
        self.evaluate_obligation_no_overflow(obligation).must_apply_modulo_regions()
    }

    /// Evaluate a given predicate, capturing overflow and propagating it back.
    fn evaluate_obligation(
        &self,
        obligation: &PredicateObligation<'tcx>,
    ) -> Result<EvaluationResult, OverflowError> {
        let mut _orig_values = OriginalQueryValues::default();

        let param_env = obligation.param_env;

        if self.next_trait_solver() {
            self.probe(|snapshot| {
                let ocx = ObligationCtxt::new(self);
                ocx.register_obligation(obligation.clone());
                let mut result = EvaluationResult::EvaluatedToOk;
                for error in ocx.select_all_or_error() {
                    if error.is_true_error() {
                        return Ok(EvaluationResult::EvaluatedToErr);
                    } else {
                        result = result.max(EvaluationResult::EvaluatedToAmbig);
                    }
                }
                if self.opaque_types_added_in_snapshot(snapshot) {
                    result = result.max(EvaluationResult::EvaluatedToOkModuloOpaqueTypes);
                } else if self.region_constraints_added_in_snapshot(snapshot) {
                    result = result.max(EvaluationResult::EvaluatedToOkModuloRegions);
                }
                Ok(result)
            })
        } else {
            let c_pred =
                self.canonicalize_query(param_env.and(obligation.predicate), &mut _orig_values);
            self.tcx.at(obligation.cause.span).evaluate_obligation(c_pred)
        }
    }

    /// Helper function that canonicalizes and runs the query. If an
    /// overflow results, we re-run it in the local context so we can
    /// report a nice error.
    fn evaluate_obligation_no_overflow(
        &self,
        obligation: &PredicateObligation<'tcx>,
    ) -> EvaluationResult {
        // Run canonical query. If overflow occurs, rerun from scratch but this time
        // in standard trait query mode so that overflow is handled appropriately
        // within `SelectionContext`.
        match self.evaluate_obligation(obligation) {
            Ok(result) => result,
            Err(OverflowError::Canonical) => {
                let mut selcx = SelectionContext::new(self);
                selcx.evaluate_root_obligation(obligation).unwrap_or_else(|r| match r {
                    OverflowError::Canonical => {
                        span_bug!(
                            obligation.cause.span,
                            "Overflow should be caught earlier in standard query mode: {:?}, {:?}",
                            obligation,
                            r,
                        )
                    }
                    OverflowError::Error(_) => EvaluationResult::EvaluatedToErr,
                })
            }
            Err(OverflowError::Error(_)) => EvaluationResult::EvaluatedToErr,
        }
    }
}
