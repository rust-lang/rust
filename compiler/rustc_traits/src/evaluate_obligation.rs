use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{ParamEnvAnd, TyCtxt};
use rustc_span::source_map::DUMMY_SP;
use rustc_trait_selection::traits::query::CanonicalPredicateGoal;
use rustc_trait_selection::traits::{
    EvaluationResult, Obligation, ObligationCause, OverflowError, SelectionContext,
};

pub(crate) fn provide(p: &mut Providers) {
    *p = Providers { evaluate_obligation, ..*p };
}

fn evaluate_obligation<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonical_goal: CanonicalPredicateGoal<'tcx>,
) -> Result<EvaluationResult, OverflowError> {
    debug!("evaluate_obligation(canonical_goal={:#?})", canonical_goal);
    let (ref infcx, goal, _canonical_inference_vars) =
        tcx.infer_ctxt().build_with_canonical(DUMMY_SP, &canonical_goal);
    debug!("evaluate_obligation: goal={:#?}", goal);
    let ParamEnvAnd { param_env, value: predicate } = goal;

    let mut selcx = SelectionContext::new_in_canonical_query(&infcx);
    let obligation = Obligation::new(tcx, ObligationCause::dummy(), param_env, predicate);

    selcx.evaluate_root_obligation(&obligation)
}
