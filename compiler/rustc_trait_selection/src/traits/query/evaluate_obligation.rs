use rustc_middle::ty;

use crate::infer::canonical::OriginalQueryValues;
use crate::infer::InferCtxt;
use crate::traits::{
    EvaluationResult, OverflowError, PredicateObligation, SelectionContext, TraitQueryMode,
};

pub trait InferCtxtExt<'tcx> {
    fn predicate_may_hold(&self, obligation: &PredicateObligation<'tcx>) -> bool;

    fn predicate_must_hold_considering_regions(
        &self,
        obligation: &PredicateObligation<'tcx>,
    ) -> bool;

    fn predicate_must_hold_modulo_regions(&self, obligation: &PredicateObligation<'tcx>) -> bool;

    fn evaluate_obligation(
        &self,
        obligation: &PredicateObligation<'tcx>,
    ) -> Result<EvaluationResult, OverflowError>;

    // Helper function that canonicalizes and runs the query. If an
    // overflow results, we re-run it in the local context so we can
    // report a nice error.
    /*crate*/
    fn evaluate_obligation_no_overflow(
        &self,
        obligation: &PredicateObligation<'tcx>,
    ) -> EvaluationResult;
}

impl<'cx, 'tcx> InferCtxtExt<'tcx> for InferCtxt<'cx, 'tcx> {
    /// Evaluates whether the predicate can be satisfied (by any means)
    /// in the given `ParamEnv`.
    fn predicate_may_hold(&self, obligation: &PredicateObligation<'tcx>) -> bool {
        self.evaluate_obligation_no_overflow(obligation).may_apply()
    }

    /// Evaluates whether the predicate can be satisfied in the given
    /// `ParamEnv`, and returns `false` if not certain. However, this is
    /// not entirely accurate if inference variables are involved.
    ///
    /// This version may conservatively fail when outlives obligations
    /// are required.
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

        let param_env = match obligation.predicate.kind().skip_binder() {
            ty::PredicateKind::Trait(pred) => {
                // we ignore the value set to it.
                let mut _constness = pred.constness;
                obligation
                    .param_env
                    .with_constness(_constness.and(obligation.param_env.constness()))
            }
            // constness has no effect on the given predicate.
            _ => obligation.param_env.without_const(),
        };

        let c_pred = self
            .canonicalize_query_keep_static(param_env.and(obligation.predicate), &mut _orig_values);
        // Run canonical query. If overflow occurs, rerun from scratch but this time
        // in standard trait query mode so that overflow is handled appropriately
        // within `SelectionContext`.
        self.tcx.at(obligation.cause.span(self.tcx)).evaluate_obligation(c_pred)
    }

    // Helper function that canonicalizes and runs the query. If an
    // overflow results, we re-run it in the local context so we can
    // report a nice error.
    fn evaluate_obligation_no_overflow(
        &self,
        obligation: &PredicateObligation<'tcx>,
    ) -> EvaluationResult {
        match self.evaluate_obligation(obligation) {
            Ok(result) => result,
            Err(OverflowError::Canonical) => {
                let mut selcx = SelectionContext::with_query_mode(&self, TraitQueryMode::Standard);
                selcx.evaluate_root_obligation(obligation).unwrap_or_else(|r| match r {
                    OverflowError::Canonical => {
                        span_bug!(
                            obligation.cause.span,
                            "Overflow should be caught earlier in standard query mode: {:?}, {:?}",
                            obligation,
                            r,
                        )
                    }
                    OverflowError::ErrorReporting => EvaluationResult::EvaluatedToErr,
                })
            }
            Err(OverflowError::ErrorReporting) => EvaluationResult::EvaluatedToErr,
        }
    }
}
