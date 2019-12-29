use rustc::infer::canonical::{Canonical, QueryResponse};
use rustc::traits::query::{CanonicalTraitGoal, NoSolution};
use rustc::traits::{Obligation, ObligationCause, SelectionContext, TraitQueryMode, Vtable};
use rustc::ty::query::Providers;
use rustc::ty::{ParamEnvAnd, TyCtxt};

crate fn provide(p: &mut Providers<'_>) {
    *p = Providers { resolve_vtable, ..*p };
}

/// Attempts to resolve a vtable.
pub fn resolve_vtable<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonical_goal: CanonicalTraitGoal<'tcx>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, Vtable<'tcx, ()>>>, NoSolution> {
    // Do the initial selection for the obligation. This yields the
    // shallow result we are looking for -- that is, what specific impl.
    tcx.infer_ctxt().enter_canonical_trait_query(
        &canonical_goal,
        |infcx, fulfill_cx, ParamEnvAnd { param_env, value: trait_ref }| {
            debug!(
                "resolve_vtable(param_env={:?}, trait_ref={:?}, def_id={:?})",
                param_env,
                trait_ref,
                trait_ref.def_id()
            );

            let mut selcx = SelectionContext::with_query_mode(&infcx, TraitQueryMode::Canonical);

            let obligation_cause = ObligationCause::dummy();
            let obligation =
                Obligation::new(obligation_cause, param_env, trait_ref.to_poly_trait_predicate());

            let selection = match selcx.select(&obligation) {
                Ok(Some(selection)) => selection,
                Ok(None) => return Err(NoSolution),
                Err(e) => {
                    debug!(
                        "Encountered error `{:?}` when resolving vtable for `{:?}`",
                        e, trait_ref
                    );
                    return Err(NoSolution);
                }
            };

            debug!("resolve_vtable: selection={:?}", selection);

            // Currently, we use a fulfillment context to completely resolve
            // all nested obligations. This is because they can inform the
            // inference of the impl's type parameters.
            let mut vtable = selection.map(|predicate| {
                debug!("resolve_vtable: register_predicate_obligation {:?}", predicate);
                fulfill_cx.register_predicate_obligation(&infcx, predicate);
            });

            vtable = infcx.resolve_vars_if_possible(&vtable);
            vtable = tcx.erase_regions(&vtable);

            info!("Cache miss: {:?} => {:?}", trait_ref, vtable);
            Ok(vtable)
        },
    )
}
