use infer::InferCtxt;
use smallvec::SmallVec;
use traits::{EvaluationResult, PredicateObligation, SelectionContext,
             TraitQueryMode, OverflowError};

impl<'cx, 'gcx, 'tcx> InferCtxt<'cx, 'gcx, 'tcx> {
    /// Evaluates whether the predicate can be satisfied (by any means)
    /// in the given `ParamEnv`.
    pub fn predicate_may_hold(
        &self,
        obligation: &PredicateObligation<'tcx>,
    ) -> bool {
        self.evaluate_obligation(obligation).may_apply()
    }

    /// Evaluates whether the predicate can be satisfied in the given
    /// `ParamEnv`, and returns `false` if not certain. However, this is
    /// not entirely accurate if inference variables are involved.
    pub fn predicate_must_hold(
        &self,
        obligation: &PredicateObligation<'tcx>,
    ) -> bool {
        self.evaluate_obligation(obligation) == EvaluationResult::EvaluatedToOk
    }

    // Helper function that canonicalizes and runs the query, as well as handles
    // overflow.
    fn evaluate_obligation(
        &self,
        obligation: &PredicateObligation<'tcx>,
    ) -> EvaluationResult {
        let mut _orig_values = SmallVec::new();
        let c_pred = self.canonicalize_query(&obligation.param_env.and(obligation.predicate),
                                             &mut _orig_values);
        // Run canonical query. If overflow occurs, rerun from scratch but this time
        // in standard trait query mode so that overflow is handled appropriately
        // within `SelectionContext`.
        match self.tcx.global_tcx().evaluate_obligation(c_pred) {
            Ok(result) => result,
            Err(OverflowError) => {
                let mut selcx =
                    SelectionContext::with_query_mode(&self, TraitQueryMode::Standard);
                selcx.evaluate_obligation_recursively(obligation)
                     .expect("Overflow should be caught earlier in standard query mode")
            }
        }
    }
}
