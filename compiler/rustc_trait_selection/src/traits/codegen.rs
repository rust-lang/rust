// This file contains various trait resolution methods used by codegen.
// They all assume regions can be erased and monomorphic types.  It
// seems likely that they should eventually be merged into more
// general routines.

use crate::infer::{InferCtxt, TyCtxtInferExt};
use crate::traits::{
    FulfillmentContext, ImplSource, Obligation, ObligationCause, SelectionContext, TraitEngine,
    Unimplemented,
};
use rustc_errors::ErrorReported;
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::{self, TyCtxt};

/// Attempts to resolve an obligation to an `ImplSource`. The result is
/// a shallow `ImplSource` resolution, meaning that we do not
/// (necessarily) resolve all nested obligations on the impl. Note
/// that type check should guarantee to us that all nested
/// obligations *could be* resolved if we wanted to.
///
/// This also expects that `trait_ref` is fully normalized.
pub fn codegen_fulfill_obligation<'tcx>(
    tcx: TyCtxt<'tcx>,
    (param_env, trait_ref): (ty::ParamEnv<'tcx>, ty::PolyTraitRef<'tcx>),
) -> Result<&'tcx ImplSource<'tcx, ()>, ErrorReported> {
    // Remove any references to regions; this helps improve caching.
    let trait_ref = tcx.erase_regions(trait_ref);
    // We expect the input to be fully normalized.
    debug_assert_eq!(trait_ref, tcx.normalize_erasing_regions(param_env, trait_ref));
    debug!(
        "codegen_fulfill_obligation(trait_ref={:?}, def_id={:?})",
        (param_env, trait_ref),
        trait_ref.def_id()
    );

    // Do the initial selection for the obligation. This yields the
    // shallow result we are looking for -- that is, what specific impl.
    tcx.infer_ctxt().enter(|infcx| {
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
                return Err(ErrorReported);
            }
            Err(Unimplemented) => {
                // This can trigger when we probe for the source of a `'static` lifetime requirement
                // on a trait object: `impl Foo for dyn Trait {}` has an implicit `'static` bound.
                // This can also trigger when we have a global bound that is not actually satisfied,
                // but was included during typeck due to the trivial_bounds feature.
                infcx.tcx.sess.delay_span_bug(
                    rustc_span::DUMMY_SP,
                    &format!(
                        "Encountered error `Unimplemented` selecting `{:?}` during codegen",
                        trait_ref
                    ),
                );
                return Err(ErrorReported);
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
        let impl_source = selection.map(|predicate| {
            debug!("fulfill_obligation: register_predicate_obligation {:?}", predicate);
            fulfill_cx.register_predicate_obligation(&infcx, predicate);
        });
        let impl_source = drain_fulfillment_cx_or_panic(&infcx, &mut fulfill_cx, impl_source);

        debug!("Cache miss: {:?} => {:?}", trait_ref, impl_source);
        Ok(&*tcx.arena.alloc(impl_source))
    })
}

// # Global Cache

/// Finishes processes any obligations that remain in the
/// fulfillment context, and then returns the result with all type
/// variables removed and regions erased. Because this is intended
/// for use outside of type inference, if any errors occur,
/// it will panic. It is used during normalization and other cases
/// where processing the obligations in `fulfill_cx` may cause
/// type inference variables that appear in `result` to be
/// unified, and hence we need to process those obligations to get
/// the complete picture of the type.
fn drain_fulfillment_cx_or_panic<'tcx, T>(
    infcx: &InferCtxt<'_, 'tcx>,
    fulfill_cx: &mut FulfillmentContext<'tcx>,
    result: T,
) -> T
where
    T: TypeFoldable<'tcx>,
{
    debug!("drain_fulfillment_cx_or_panic()");

    // In principle, we only need to do this so long as `result`
    // contains unbound type parameters. It could be a slight
    // optimization to stop iterating early.
    let errors = fulfill_cx.select_all_or_error(infcx);
    if !errors.is_empty() {
        infcx.tcx.sess.delay_span_bug(
            rustc_span::DUMMY_SP,
            &format!(
                "Encountered errors `{:?}` resolving bounds outside of type inference",
                errors
            ),
        );
    }

    let result = infcx.resolve_vars_if_possible(result);
    infcx.tcx.erase_regions(result)
}
