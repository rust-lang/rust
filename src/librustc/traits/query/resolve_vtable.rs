use crate::infer::canonical::OriginalQueryValues;
use crate::infer::InferCtxt;
use crate::traits::{ObligationCause, Vtable};
use crate::ty;

impl<'cx, 'tcx> InferCtxt<'cx, 'tcx> {
    /// Attempts to resolves the `Vtable` for a given trait within a `ParamEnv`.
    ///
    /// If it can't due to the result being ambiguous, or an error occurred during selection, `None`
    /// is returned.
    pub fn resolve_vtable(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        trait_ref: ty::TraitRef<'tcx>,
    ) -> Option<Vtable<'tcx, ()>> {
        // Remove any references to regions; this helps improve caching.
        let trait_ref = self.tcx.erase_regions(&trait_ref);

        let mut orig_values = OriginalQueryValues::default();
        let canonical =
            self.canonicalize_query(&param_env.and(ty::Binder::bind(trait_ref)), &mut orig_values);
        if let Ok(query_response) = self.tcx.resolve_vtable(canonical) {
            if query_response.is_proven() {
                let result = self.instantiate_query_response_and_region_obligations(
                    &ObligationCause::dummy(),
                    param_env,
                    &orig_values,
                    query_response,
                );
                return Some(result.unwrap().value);
            }
        };
        None
    }
}
