use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, ParamEnvAnd, TyCtxt, TypeFoldable};
use rustc_trait_selection::traits::query::normalize::AtExt;
use rustc_trait_selection::traits::{Normalized, ObligationCause};
use std::sync::atomic::Ordering;

crate fn provide(p: &mut Providers) {
    *p = Providers {
        try_normalize_generic_arg_after_erasing_regions: |tcx, goal| {
            debug!("try_normalize_generic_arg_after_erasing_regions(goal={:#?}", goal);

            tcx.sess
                .perf_stats
                .normalize_generic_arg_after_erasing_regions
                .fetch_add(1, Ordering::Relaxed);

            try_normalize_after_erasing_regions(tcx, goal)
        },
        try_normalize_mir_const_after_erasing_regions: |tcx, goal| {
            try_normalize_after_erasing_regions(tcx, goal)
        },
        ..*p
    };
}

#[instrument(level = "debug", skip(tcx))]
fn try_normalize_after_erasing_regions<'tcx, T: TypeFoldable<'tcx> + PartialEq + Copy>(
    tcx: TyCtxt<'tcx>,
    goal: ParamEnvAnd<'tcx, T>,
) -> Result<T, NoSolution> {
    let ParamEnvAnd { param_env, value } = goal;
    tcx.infer_ctxt().enter(|infcx| {
        let cause = ObligationCause::dummy();
        match infcx.at(&cause, param_env).normalize(value) {
            Ok(Normalized { value: normalized_value, obligations: normalized_obligations }) => {
                // We don't care about the `obligations`; they are
                // always only region relations, and we are about to
                // erase those anyway:
                debug_assert_eq!(
                    normalized_obligations.iter().find(|p| not_outlives_predicate(p.predicate)),
                    None,
                );

                let resolved_value = infcx.resolve_vars_if_possible(normalized_value);
                // It's unclear when `resolve_vars` would have an effect in a
                // fresh `InferCtxt`. If this assert does trigger, it will give
                // us a test case.
                debug_assert_eq!(normalized_value, resolved_value);
                let erased = infcx.tcx.erase_regions(resolved_value);
                debug_assert!(!erased.needs_infer(), "{:?}", erased);
                Ok(erased)
            }
            Err(NoSolution) => Err(NoSolution),
        }
    })
}

fn not_outlives_predicate<'tcx>(p: ty::Predicate<'tcx>) -> bool {
    match p.kind().skip_binder() {
        ty::PredicateKind::RegionOutlives(..) | ty::PredicateKind::TypeOutlives(..) => false,
        ty::PredicateKind::Trait(..)
        | ty::PredicateKind::Projection(..)
        | ty::PredicateKind::WellFormed(..)
        | ty::PredicateKind::ObjectSafe(..)
        | ty::PredicateKind::ClosureKind(..)
        | ty::PredicateKind::Subtype(..)
        | ty::PredicateKind::Coerce(..)
        | ty::PredicateKind::ConstEvaluatable(..)
        | ty::PredicateKind::ConstEquate(..)
        | ty::PredicateKind::TypeWellFormedFromEnv(..) => true,
    }
}
