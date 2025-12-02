use rustc_hir::def_id::LocalDefId;
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::{InferCtxt, RegionResolutionError};
use rustc_macros::extension;
use rustc_middle::traits::ObligationCause;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::{self, Ty, elaborate};

use crate::traits::ScrubbedTraitError;
use crate::traits::outlives_bounds::InferCtxtExt;

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

        // FIXME(-Znext-trait-solver): Normalize these.
        let higher_ranked_assumptions = infcx.take_registered_region_assumptions();
        let higher_ranked_assumptions =
            elaborate::elaborate_outlives_assumptions(infcx.tcx, higher_ranked_assumptions);

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
    /// Resolve regions, using the deep normalizer to normalize any type-outlives
    /// obligations in the process. This is in `rustc_trait_selection` because
    /// we need to normalize.
    ///
    /// Prefer this method over `resolve_regions_with_normalize`, unless you are
    /// doing something specific for normalization.
    ///
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

    /// Don't call this directly unless you know what you're doing.
    fn resolve_regions_with_outlives_env(
        &self,
        outlives_env: &OutlivesEnvironment<'tcx>,
    ) -> Vec<RegionResolutionError<'tcx>> {
        self.resolve_regions_with_normalize(&outlives_env, |ty, origin| {
            let ty = self.resolve_vars_if_possible(ty);

            if self.next_trait_solver() {
                crate::solve::deeply_normalize(
                    self.at(
                        &ObligationCause::dummy_with_span(origin.span()),
                        outlives_env.param_env,
                    ),
                    ty,
                )
                .map_err(|_: Vec<ScrubbedTraitError<'tcx>>| NoSolution)
            } else {
                Ok(ty)
            }
        })
    }
}
