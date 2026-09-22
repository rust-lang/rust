//! Checks whether an output depends only on the complete types of its inputs.

use rustc_data_structures::fx::FxIndexSet;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt, TypingMode, Unnormalized};

use super::{ObligationCause, ObligationCtxt, ScrubbedTraitError};
use crate::regions::InferCtxtRegionExt;

/// Inputs and output share a binder. In particular, normalization must not
/// instantiate their bound variables independently.
pub type OutputTypeDependency<'tcx> = ty::Binder<'tcx, (ty::GenericArgsRef<'tcx>, ty::Term<'tcx>)>;

pub fn unconstrained_output_regions<'tcx>(
    tcx: TyCtxt<'tcx>,
    dependency: OutputTypeDependency<'tcx>,
) -> FxIndexSet<ty::BoundRegionKind<'tcx>> {
    let inputs = dependency.map_bound(|(inputs, _)| inputs);
    let constrained = tcx.collect_constrained_late_bound_regions(inputs);
    let referenced = tcx.collect_output_late_bound_regions(
        inputs.map_bound(|inputs| inputs.types().collect()),
        dependency.map_bound(|(_, output)| output),
    );
    referenced.difference(&constrained).copied().collect()
}

pub fn projection_output_dependency<'tcx>(
    projection: ty::PolyProjectionClause<'tcx>,
) -> OutputTypeDependency<'tcx> {
    projection.map_bound(|projection| (projection.projection_term.args, projection.term))
}

/// Build the environment before normalizing it, so that a pending dependency
/// check cannot use its own equality as a proof. Filtering an already normalized
/// environment is insufficient: normalization can change both sides of a clause.
pub fn output_dependency_param_env<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> ty::ParamEnv<'tcx> {
    let ignore_equalities =
        tcx.def_kind(def_id) == DefKind::TyAlias && !tcx.type_alias_is_checked(def_id);
    let clauses = tcx.clauses_of(def_id).instantiate_identity(tcx);
    let clauses =
        super::elaborate(tcx, clauses.clauses.into_iter().map(Unnormalized::skip_norm_wip));
    ty::ParamEnv::new(
        tcx,
        clauses.filter(|clause| {
            let Some(projection) = clause.as_projection_clause() else { return true };
            !ignore_equalities
                && unconstrained_output_regions(tcx, projection_output_dependency(projection))
                    .is_empty()
        }),
    )
}

/// Refine a structural dependency check using independently available equalities.
///
/// This uses a separate inference context that checks regions, including when
/// called from HIR type checking (whose inference context ignores regions).
/// Neither a failed normalization nor an unsatisfied region constraint supplies
/// evidence that the output is determined by the inputs.
pub fn unconstrained_output_regions_after_normalization<'tcx>(
    tcx: TyCtxt<'tcx>,
    cause: &ObligationCause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    dependency: OutputTypeDependency<'tcx>,
) -> FxIndexSet<ty::BoundRegionKind<'tcx>> {
    let original = unconstrained_output_regions(tcx, dependency);
    if original.is_empty() || !dependency.has_aliases() || dependency.has_infer() {
        return original;
    }

    let infcx =
        tcx.infer_ctxt().with_next_trait_solver(true).build(TypingMode::non_body_analysis());
    let ocx = ObligationCtxt::new(&infcx);
    let Ok(clauses) = ocx.deeply_normalize(
        cause,
        param_env,
        Unnormalized::new_wip(param_env.caller_bounds().collect::<Vec<_>>()),
    ) else {
        return original;
    };
    let param_env = super::outlives_bounds::elaborate_projection_outlives(
        tcx,
        cause,
        ty::ParamEnv::new(tcx, clauses),
    );
    let mut universes = Vec::new();
    while dependency.has_vars_bound_at_or_above(ty::DebruijnIndex::from_usize(universes.len())) {
        universes.push(None);
    }
    let normalized = crate::solve::deeply_normalize_with_skipped_universes::<
        _,
        ScrubbedTraitError<'tcx>,
    >(infcx.at(cause, param_env), Unnormalized::new_wip(dependency), universes);
    let Ok(normalized) = normalized else { return original };
    if normalized.has_infer() || !infcx.resolve_regions(cause.body_def_id, param_env, []).is_empty()
    {
        return original;
    }

    let remaining = unconstrained_output_regions(tcx, normalized);
    original.intersection(&remaining).copied().collect()
}
