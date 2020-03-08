// This file contains various trait resolution methods used by codegen.
// They all assume regions can be erased and monomorphic types.  It
// seems likely that they should eventually be merged into more
// general routines.

use crate::infer::{InferCtxt, TyCtxtInferExt};
use crate::traits::{
    FulfillmentContext, Obligation, ObligationCause, SelectionContext, TraitEngine, Vtable,
};
use rustc::ty::fold::TypeFoldable;
use rustc::ty::{self, TyCtxt};

/// Attempts to resolve an obligation to a vtable. The result is
/// a shallow vtable resolution, meaning that we do not
/// (necessarily) resolve all nested obligations on the impl. Note
/// that type check should guarantee to us that all nested
/// obligations *could be* resolved if we wanted to.
/// Assumes that this is run after the entire crate has been successfully type-checked.
pub fn codegen_fulfill_obligation<'tcx>(
    ty: TyCtxt<'tcx>,
    (param_env, trait_ref): (ty::ParamEnv<'tcx>, ty::PolyTraitRef<'tcx>),
) -> Option<Vtable<'tcx, ()>> {
    // Remove any references to regions; this helps improve caching.
    let trait_ref = ty.erase_regions(&trait_ref);

    debug!(
        "codegen_fulfill_obligation(trait_ref={:?}, def_id={:?})",
        (param_env, trait_ref),
        trait_ref.def_id()
    );

    // Do the initial selection for the obligation. This yields the
    // shallow result we are looking for -- that is, what specific impl.
    ty.infer_ctxt().enter(|infcx| {
        let mut selcx = SelectionContext::new(&infcx);

        let obligation_cause = ObligationCause::dummy();
        let obligation =
            Obligation::new(obligation_cause, param_env, trait_ref.to_poly_trait_predicate());

        let selection = match selcx.select(&obligation) {
            Ok(Some(selection)) => selection,
            Ok(None) => {
                // Ambiguity can happen when monomorphizing during trans
                // expands to some humongo type that never occurred
                // statically -- this humongo type can then overflow,
                // leading to an ambiguous result. So report this as an
                // overflow bug, since I believe this is the only case
                // where ambiguity can result.
                infcx.tcx.sess.delay_span_bug(
                    rustc_span::DUMMY_SP,
                    &format!(
                        "encountered ambiguity selecting `{:?}` during codegen, presuming due to \
                         overflow or prior type error",
                        trait_ref
                    ),
                );
                return None;
            }
            Err(e) => {
                bug!("Encountered error `{:?}` selecting `{:?}` during codegen", e, trait_ref)
            }
        };

        debug!("fulfill_obligation: selection={:?}", selection);

        // Currently, we use a fulfillment context to completely resolve
        // all nested obligations. This is because they can inform the
        // inference of the impl's type parameters.
        let mut fulfill_cx = FulfillmentContext::new();
        let vtable = selection.map(|predicate| {
            debug!("fulfill_obligation: register_predicate_obligation {:?}", predicate);
            fulfill_cx.register_predicate_obligation(&infcx, predicate);
        });
        let vtable = infcx.drain_fulfillment_cx_or_panic(&mut fulfill_cx, &vtable);

        info!("Cache miss: {:?} => {:?}", trait_ref, vtable);
        Some(vtable)
    })
}

// # Global Cache

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    /// Finishes processes any obligations that remain in the
    /// fulfillment context, and then returns the result with all type
    /// variables removed and regions erased. Because this is intended
    /// for use after type-check has completed, if any errors occur,
    /// it will panic. It is used during normalization and other cases
    /// where processing the obligations in `fulfill_cx` may cause
    /// type inference variables that appear in `result` to be
    /// unified, and hence we need to process those obligations to get
    /// the complete picture of the type.
    fn drain_fulfillment_cx_or_panic<T>(
        &self,
        fulfill_cx: &mut FulfillmentContext<'tcx>,
        result: &T,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        debug!("drain_fulfillment_cx_or_panic()");

        // In principle, we only need to do this so long as `result`
        // contains unbound type parameters. It could be a slight
        // optimization to stop iterating early.
        if let Err(errors) = fulfill_cx.select_all_or_error(self) {
            bug!("Encountered errors `{:?}` resolving bounds after type-checking", errors);
        }

        let result = self.resolve_vars_if_possible(result);
        self.tcx.erase_regions(&result)
    }
}
