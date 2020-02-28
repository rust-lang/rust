use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::subst::GenericArg;
use rustc_middle::ty::{self, ParamEnvAnd, TyCtxt};
use rustc_trait_selection::traits::query::normalize::AtExt;
use rustc_trait_selection::traits::{Normalized, ObligationCause};
use std::sync::atomic::Ordering;

crate fn provide(p: &mut Providers<'_>) {
    *p = Providers { normalize_generic_arg_after_erasing_regions, ..*p };
}

fn normalize_generic_arg_after_erasing_regions<'tcx>(
    tcx: TyCtxt<'tcx>,
    goal: ParamEnvAnd<'tcx, GenericArg<'tcx>>,
) -> GenericArg<'tcx> {
    debug!("normalize_generic_arg_after_erasing_regions(goal={:#?})", goal);

    let ParamEnvAnd { param_env, value } = goal;
    tcx.sess.perf_stats.normalize_generic_arg_after_erasing_regions.fetch_add(1, Ordering::Relaxed);
    tcx.infer_ctxt().enter(|infcx| {
        let cause = ObligationCause::dummy();
        match infcx.at(&cause, param_env).normalize(&value) {
            Ok(Normalized { value: normalized_value, obligations: normalized_obligations }) => {
                // We don't care about the `obligations`; they are
                // always only region relations, and we are about to
                // erase those anyway:
                debug_assert_eq!(
                    normalized_obligations.iter().find(|p| not_outlives_predicate(&p.predicate)),
                    None,
                );

                let normalized_value = infcx.resolve_vars_if_possible(&normalized_value);
                infcx.tcx.erase_regions(&normalized_value)
            }
            Err(NoSolution) => bug!("could not fully normalize `{:?}`", value),
        }
    })
}

fn not_outlives_predicate(p: &ty::Predicate<'_>) -> bool {
    match p {
        ty::Predicate::RegionOutlives(..) | ty::Predicate::TypeOutlives(..) => false,
        ty::Predicate::Trait(..)
        | ty::Predicate::Projection(..)
        | ty::Predicate::WellFormed(..)
        | ty::Predicate::ObjectSafe(..)
        | ty::Predicate::ClosureKind(..)
        | ty::Predicate::Subtype(..)
        | ty::Predicate::ConstEvaluatable(..)
        | ty::Predicate::ConstEquate(..) => true,
    }
}
