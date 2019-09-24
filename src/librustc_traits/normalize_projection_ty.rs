use rustc::hir;
use rustc::infer::canonical::{Canonical, QueryResponse};
use rustc::traits::query::{normalize::NormalizationResult, CanonicalProjectionGoal, NoSolution};
use rustc::traits::{self, ObligationCause, SelectionContext, TraitEngineExt};
use rustc::ty::query::Providers;
use rustc::ty::{ParamEnvAnd, TyCtxt};
use std::sync::atomic::Ordering;
use syntax_pos::DUMMY_SP;

crate fn provide(p: &mut Providers<'_>) {
    *p = Providers {
        normalize_projection_ty,
        ..*p
    };
}

fn normalize_projection_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    goal: CanonicalProjectionGoal<'tcx>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, NormalizationResult<'tcx>>>, NoSolution> {
    debug!("normalize_provider(goal={:#?})", goal);

    tcx.sess
        .perf_stats
        .normalize_projection_ty
        .fetch_add(1, Ordering::Relaxed);
    tcx.infer_ctxt().enter_canonical_trait_query(
        &goal,
        |infcx,
         fulfill_cx,
         ParamEnvAnd {
             param_env,
             value: goal,
         }| {
            let selcx = &mut SelectionContext::new(infcx);
            let cause = ObligationCause::misc(DUMMY_SP, hir::DUMMY_HIR_ID);
            let mut obligations = vec![];
            let answer = traits::normalize_projection_type(
                selcx,
                param_env,
                goal,
                cause,
                0,
                &mut obligations,
            );
            fulfill_cx.register_predicate_obligations(infcx, obligations);
            Ok(NormalizationResult {
                normalized_ty: answer,
            })
        },
    )
}
