use rustc::ty::query::Providers;
use rustc::ty::{ParamEnvAnd, TyCtxt};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_span::source_map::DUMMY_SP;
use rustc_trait_selection::traits::query::CanonicalPredicateGoal;
use rustc_trait_selection::traits::{
    EvaluationResult, Obligation, ObligationCause, OverflowError, SelectionContext, TraitQueryMode,
};

crate fn provide(p: &mut Providers<'_>) {
    *p = Providers { evaluate_obligation, ..*p };
}

fn evaluate_obligation<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonical_goal: CanonicalPredicateGoal<'tcx>,
) -> Result<EvaluationResult, OverflowError> {
    debug!("evaluate_obligation(canonical_goal={:#?})", canonical_goal);
    tcx.infer_ctxt().enter_with_canonical(
        DUMMY_SP,
        &canonical_goal,
        |ref infcx, goal, _canonical_inference_vars| {
            debug!("evaluate_obligation: goal={:#?}", goal);
            let ParamEnvAnd { param_env, value: predicate } = goal;

            let mut selcx = SelectionContext::with_query_mode(&infcx, TraitQueryMode::Canonical);
            let obligation = Obligation::new(ObligationCause::dummy(), param_env, predicate);

            selcx.evaluate_root_obligation(&obligation)
        },
    )
}
