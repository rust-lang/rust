use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::query::Providers;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::{self, PseudoCanonicalInput, TyCtxt, TypeFoldable, TypeVisitableExt};
use rustc_trait_selection::traits::query::normalize::QueryNormalizeExt;
use rustc_trait_selection::traits::{Normalized, ObligationCause};
use tracing::debug;

pub(crate) fn provide(p: &mut Providers) {
    *p = Providers {
        try_normalize_generic_arg_after_erasing_regions: |tcx, goal| {
            debug!("try_normalize_generic_arg_after_erasing_regions(goal={:#?}", goal);

            try_normalize_after_erasing_regions(tcx, goal)
        },
        ..*p
    };
}

fn try_normalize_after_erasing_regions<'tcx, T: TypeFoldable<TyCtxt<'tcx>> + PartialEq + Copy>(
    tcx: TyCtxt<'tcx>,
    goal: PseudoCanonicalInput<'tcx, T>,
) -> Result<T, NoSolution> {
    let PseudoCanonicalInput { typing_env, value } = goal;
    let (infcx, param_env) = tcx.infer_ctxt().build_with_typing_env(typing_env);
    let cause = ObligationCause::dummy();
    match infcx.at(&cause, param_env).query_normalize(value) {
        Ok(Normalized { value: normalized_value, obligations: normalized_obligations }) => {
            // We don't care about the `obligations`; they are
            // always only region relations, and we are about to
            // erase those anyway:
            // This has been seen to fail in RL, so making it a non-debug assertion to better catch
            // those cases.
            assert_eq!(
                normalized_obligations.iter().find(|p| not_outlives_predicate(p.predicate)),
                None,
            );

            let resolved_value = infcx.resolve_vars_if_possible(normalized_value);
            // It's unclear when `resolve_vars` would have an effect in a
            // fresh `InferCtxt`. If this assert does trigger, it will give
            // us a test case.
            debug_assert_eq!(normalized_value, resolved_value);
            let erased = infcx.tcx.erase_and_anonymize_regions(resolved_value);
            debug_assert!(!erased.has_infer(), "{erased:?}");
            Ok(erased)
        }
        Err(NoSolution) => Err(NoSolution),
    }
}

fn not_outlives_predicate(p: ty::Predicate<'_>) -> bool {
    match p.kind().skip_binder() {
        ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(..))
        | ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(..)) => false,
        ty::PredicateKind::Clause(ty::ClauseKind::Trait(..))
        | ty::PredicateKind::Clause(ty::ClauseKind::Projection(..))
        | ty::PredicateKind::Clause(ty::ClauseKind::HostEffect(..))
        | ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(..))
        | ty::PredicateKind::Clause(ty::ClauseKind::UnstableFeature(_))
        | ty::PredicateKind::NormalizesTo(..)
        | ty::PredicateKind::AliasRelate(..)
        | ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(..))
        | ty::PredicateKind::DynCompatible(..)
        | ty::PredicateKind::Subtype(..)
        | ty::PredicateKind::Coerce(..)
        | ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(..))
        | ty::PredicateKind::ConstEquate(..)
        | ty::PredicateKind::Ambiguous => true,
    }
}
