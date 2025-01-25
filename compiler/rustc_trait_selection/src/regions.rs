use rustc_hir::def_id::LocalDefId;
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::{InferCtxt, RegionResolutionError};
use rustc_macros::extension;
use rustc_middle::traits::ObligationCause;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::{self, Ty};

use crate::traits::ScrubbedTraitError;
use crate::traits::outlives_bounds::InferCtxtExt;

#[extension(pub trait InferCtxtRegionExt<'tcx>)]
impl<'tcx> InferCtxt<'tcx> {
    /// Resolve regions, using the deep normalizer to normalize any type-outlives
    /// obligations in the process. This is in `rustc_trait_selection` because
    /// we need to normalize.
    ///
    /// Prefer this method over `resolve_regions_with_normalize`, unless you are
    /// doing something specific for normalization.
    fn resolve_regions(
        &self,
        body_id: LocalDefId,
        param_env: ty::ParamEnv<'tcx>,
        assumed_wf_tys: impl IntoIterator<Item = Ty<'tcx>>,
    ) -> Vec<RegionResolutionError<'tcx>> {
        self.resolve_regions_with_outlives_env(&OutlivesEnvironment::with_bounds(
            param_env,
            self.implied_bounds_tys(body_id, param_env, assumed_wf_tys),
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
