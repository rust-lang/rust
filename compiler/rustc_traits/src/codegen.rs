// This file contains various trait resolution methods used by codegen.
// They all assume regions can be erased and monomorphic types. It
// seems likely that they should eventually be merged into more
// general routines.

use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::{FulfillmentErrorCode, TraitEngineExt as _};
use rustc_middle::traits::CodegenObligationError;
use rustc_middle::ty::{self, TyCtxt};
use rustc_trait_selection::traits::error_reporting::TypeErrCtxtExt;
use rustc_trait_selection::traits::{
    ImplSource, Obligation, ObligationCause, SelectionContext, TraitEngine, TraitEngineExt,
    Unimplemented,
};

/// Attempts to resolve an obligation to an `ImplSource`. The result is
/// a shallow `ImplSource` resolution, meaning that we do not
/// (necessarily) resolve all nested obligations on the impl. Note
/// that type check should guarantee to us that all nested
/// obligations *could be* resolved if we wanted to.
///
/// This also expects that `trait_ref` is fully normalized.
pub fn codegen_select_candidate<'tcx>(
    tcx: TyCtxt<'tcx>,
    (param_env, trait_ref): (ty::ParamEnv<'tcx>, ty::TraitRef<'tcx>),
) -> Result<&'tcx ImplSource<'tcx, ()>, CodegenObligationError> {
    // We expect the input to be fully normalized.
    debug_assert_eq!(trait_ref, tcx.normalize_erasing_regions(param_env, trait_ref));

    // Do the initial selection for the obligation. This yields the
    // shallow result we are looking for -- that is, what specific impl.
    let infcx = tcx.infer_ctxt().ignoring_regions().build();
    let mut selcx = SelectionContext::new(&infcx);

    let obligation_cause = ObligationCause::dummy();
    let obligation = Obligation::new(tcx, obligation_cause, param_env, trait_ref);

    let selection = match selcx.select(&obligation) {
        Ok(Some(selection)) => selection,
        Ok(None) => return Err(CodegenObligationError::Ambiguity),
        Err(Unimplemented) => return Err(CodegenObligationError::Unimplemented),
        Err(e) => {
            bug!("Encountered error `{:?}` selecting `{:?}` during codegen", e, trait_ref)
        }
    };

    debug!(?selection);

    // Currently, we use a fulfillment context to completely resolve
    // all nested obligations. This is because they can inform the
    // inference of the impl's type parameters.
    let mut fulfill_cx = <dyn TraitEngine<'tcx>>::new(&infcx);
    let impl_source = selection.map(|predicate| {
        fulfill_cx.register_predicate_obligation(&infcx, predicate);
    });

    // In principle, we only need to do this so long as `impl_source`
    // contains unbound type parameters. It could be a slight
    // optimization to stop iterating early.
    let errors = fulfill_cx.select_all_or_error(&infcx);
    if !errors.is_empty() {
        // `rustc_monomorphize::collector` assumes there are no type errors.
        // Cycle errors are the only post-monomorphization errors possible; emit them now so
        // `rustc_ty_utils::resolve_associated_item` doesn't return `None` post-monomorphization.
        for err in errors {
            if let FulfillmentErrorCode::CodeCycle(cycle) = err.code {
                infcx.err_ctxt().report_overflow_obligation_cycle(&cycle);
            }
        }
        return Err(CodegenObligationError::FulfillmentError);
    }

    let impl_source = infcx.resolve_vars_if_possible(impl_source);
    let impl_source = infcx.tcx.erase_regions(impl_source);

    Ok(&*tcx.arena.alloc(impl_source))
}
