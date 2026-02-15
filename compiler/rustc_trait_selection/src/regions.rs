use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::LocalDefId;
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::{InferCtxt, RegionResolutionError};
use rustc_macros::extension;
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::{self, Ty, elaborate};

use crate::traits::ScrubbedTraitError;
use crate::traits::outlives_bounds::InferCtxtExt;

fn normalize_higher_ranked_assumptions<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> FxHashSet<ty::ArgOutlivesPredicate<'tcx>> {
    let assumptions = infcx.take_registered_region_assumptions();
    if !infcx.next_trait_solver() {
        return elaborate::elaborate_outlives_assumptions(infcx.tcx, assumptions);
    }

    let mut normalized_assumptions = vec![];
    let mut seen_assumptions = FxHashSet::default();

    for assumption in assumptions {
        if !seen_assumptions.insert(assumption) {
            continue;
        }

        let assumption = infcx.resolve_vars_if_possible(assumption);
        let outlives = ty::Binder::dummy(assumption);
        let ty::OutlivesPredicate(kind, region) =
            match crate::solve::deeply_normalize::<_, ScrubbedTraitError<'tcx>>(
                infcx.at(&ObligationCause::dummy(), param_env),
                outlives,
            ) {
                Ok(assumption) => assumption,
                Err(_) => {
                    infcx.dcx().delayed_bug(format!(
                        "could not normalize higher-ranked assumption `{assumption}`"
                    ));
                    outlives
                }
            }
            .no_bound_vars()
            .expect("started with no bound vars, should end with no bound vars");

        normalized_assumptions.push(ty::OutlivesPredicate(kind, region));
    }

    for assumption in infcx.take_registered_region_assumptions() {
        if seen_assumptions.insert(assumption) {
            normalized_assumptions.push(assumption);
        }
    }

    elaborate::elaborate_outlives_assumptions(infcx.tcx, normalized_assumptions)
}

#[extension(pub trait OutlivesEnvironmentBuildExt<'tcx>)]
impl<'tcx> OutlivesEnvironment<'tcx> {
    fn new(
        infcx: &InferCtxt<'tcx>,
        body_id: LocalDefId,
        param_env: ty::ParamEnv<'tcx>,
        assumed_wf_tys: impl IntoIterator<Item = Ty<'tcx>>,
    ) -> Self {
        Self::new_with_implied_bounds_compat(infcx, body_id, param_env, assumed_wf_tys, false)
    }

    fn new_with_implied_bounds_compat(
        infcx: &InferCtxt<'tcx>,
        body_id: LocalDefId,
        param_env: ty::ParamEnv<'tcx>,
        assumed_wf_tys: impl IntoIterator<Item = Ty<'tcx>>,
        disable_implied_bounds_hack: bool,
    ) -> Self {
        let mut bounds = vec![];

        for bound in param_env.caller_bounds() {
            if let Some(mut type_outlives) = bound.as_type_outlives_clause() {
                if infcx.next_trait_solver() {
                    match crate::solve::deeply_normalize::<_, ScrubbedTraitError<'tcx>>(
                        infcx.at(&ObligationCause::dummy(), param_env),
                        type_outlives,
                    ) {
                        Ok(new) => type_outlives = new,
                        Err(_) => {
                            infcx.dcx().delayed_bug(format!("could not normalize `{bound}`"));
                        }
                    }
                }
                bounds.push(type_outlives);
            }
        }

        let higher_ranked_assumptions = normalize_higher_ranked_assumptions(infcx, param_env);

        // FIXME: This needs to be modified so that we normalize the known type
        // outlives obligations then elaborate them into their region/type components.
        // Otherwise, `<W<'a> as Mirror>::Assoc: 'b` will not imply `'a: 'b` even
        // if we can normalize `'a`.
        OutlivesEnvironment::from_normalized_bounds(
            param_env,
            bounds,
            infcx.implied_bounds_tys(
                body_id,
                param_env,
                assumed_wf_tys,
                disable_implied_bounds_hack,
            ),
            higher_ranked_assumptions,
        )
    }
}

#[extension(pub trait InferCtxtRegionExt<'tcx>)]
impl<'tcx> InferCtxt<'tcx> {
    /// Resolve regions.
    /// This function assumes that all infer variables are already constrained.
    fn resolve_regions(
        &self,
        body_id: LocalDefId,
        param_env: ty::ParamEnv<'tcx>,
        assumed_wf_tys: impl IntoIterator<Item = Ty<'tcx>>,
    ) -> Vec<RegionResolutionError<'tcx>> {
        self.resolve_regions_with_outlives_env(&OutlivesEnvironment::new(
            self,
            body_id,
            param_env,
            assumed_wf_tys,
        ))
    }
}
