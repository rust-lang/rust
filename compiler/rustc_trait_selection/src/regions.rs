use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::{InferCtxt, RegionResolutionError};
use rustc_macros::extension;
use rustc_middle::traits::ObligationCause;
use rustc_middle::traits::query::NoSolution;

use crate::traits::ScrubbedTraitError;

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
        outlives_env: &OutlivesEnvironment<'tcx>,
    ) -> Vec<RegionResolutionError<'tcx>> {
        self.resolve_regions_with_normalize(outlives_env, |ty, origin| {
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
