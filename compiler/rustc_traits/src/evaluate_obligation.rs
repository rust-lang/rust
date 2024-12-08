use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::query::Providers;
use rustc_middle::ty::{ParamEnvAnd, TyCtxt};
use rustc_span::DUMMY_SP;
use rustc_trait_selection::traits::query::CanonicalPredicateGoal;
use rustc_trait_selection::traits::{
    EvaluationResult, Obligation, ObligationCause, OverflowError, SelectionContext, TraitQueryMode,
};
use tracing::debug;

pub(crate) fn provide(p: &mut Providers) {
    *p = Providers { evaluate_obligation, ..*p };
}

fn evaluate_obligation<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonical_goal: CanonicalPredicateGoal<'tcx>,
) -> Result<EvaluationResult, OverflowError> {
    assert!(!tcx.next_trait_solver_globally());
    debug!("evaluate_obligation(canonical_goal={:#?})", canonical_goal);
    let (ref infcx, goal, _canonical_inference_vars) =
        tcx.infer_ctxt().build_with_canonical(DUMMY_SP, &canonical_goal);
    debug!("evaluate_obligation: goal={:#?}", goal);
    let ParamEnvAnd { param_env, value: predicate } = goal;

    let mut selcx = SelectionContext::with_query_mode(infcx, TraitQueryMode::Canonical);
    let obligation = Obligation::new(tcx, ObligationCause::dummy(), param_env, predicate);

    selcx.evaluate_root_obligation(&obligation)
}
