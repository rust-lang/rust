// This file contains various trait resolution methods used by codegen.
// They all assume regions can be erased and monomorphic types. It
// seems likely that they should eventually be merged into more
// general routines.

use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::bug;
use rustc_middle::traits::CodegenObligationError;
use rustc_middle::ty::{self, PseudoCanonicalInput, TyCtxt, TypeVisitableExt};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::traits::{
    ImplSource, Obligation, ObligationCause, ObligationCtxt, ScrubbedTraitError, SelectionContext,
    Unimplemented,
};
use tracing::{debug, warn};

/// Attempts to resolve an obligation to an `ImplSource`. The result is
/// a shallow `ImplSource` resolution, meaning that we do not
/// (necessarily) resolve all nested obligations on the impl. Note
/// that type check should guarantee to us that all nested
/// obligations *could be* resolved if we wanted to.
///
/// This also expects that `trait_ref` is fully normalized.
pub(crate) fn codegen_select_candidate<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: PseudoCanonicalInput<'tcx, ty::TraitRef<'tcx>>,
) -> Result<&'tcx ImplSource<'tcx, ()>, CodegenObligationError> {
    let PseudoCanonicalInput { typing_env, value: trait_ref } = key;

    // Ensure the input is fully normalized.
    debug_assert_eq!(trait_ref, tcx.normalize_erasing_regions(typing_env, trait_ref));

    // Do the initial selection for the obligation. This yields the
    // shallow result we are looking for -- that is, what specific impl.
    let (infcx, param_env) = tcx.infer_ctxt().ignoring_regions().build_with_typing_env(typing_env);
    let mut selcx = SelectionContext::new(&infcx);

    let obligation_cause = ObligationCause::dummy();
    let obligation = Obligation::new(tcx, obligation_cause, param_env, trait_ref);

    let selection = match selcx.select(&obligation) {
        Ok(Some(selection)) => selection,
        Ok(None) => {
            warn!(
                "Ambiguity encountered in trait resolution for {:?}. Ensure that all associated types are fully resolved.",
                trait_ref
            );
            return Err(CodegenObligationError::Ambiguity);
        }
        Err(Unimplemented) => {
            warn!(
                "Trait {:?} is not implemented for the given context during codegen.",
                trait_ref
            );
            return Err(CodegenObligationError::Unimplemented);
        }
        Err(e) => {
            bug!(
                "Unexpected error `{:?}` encountered while selecting `{:?}` during codegen.",
                e, trait_ref
            );
        }
    };

    debug!(?selection, "Selection result before resolving nested obligations");

    // Resolve all nested obligations to finalize inference of type parameters.
    let ocx = ObligationCtxt::new(&infcx);
    let impl_source = selection.map(|obligation| {
        ocx.register_obligation(obligation);
    });

    // In principle, we only need to do this so long as `impl_source`
    // contains unbound type parameters. It could be a slight
    // optimization to stop iterating early.
    let errors = ocx.select_all_or_error();
    if !errors.is_empty() {
        // `rustc_monomorphize::collector` assumes there are no type errors.
        // Cycle errors are the only post-monomorphization errors possible; emit them now so
        // `rustc_ty_utils::resolve_associated_item` doesn't return `None` post-monomorphization.
        for err in errors {
            if let ScrubbedTraitError::Cycle(cycle) = err {
                infcx.err_ctxt().report_overflow_obligation_cycle(&cycle);
            }
        }
        return Err(CodegenObligationError::FulfillmentError);
    }

    let impl_source = infcx.resolve_vars_if_possible(impl_source);
    let impl_source = tcx.erase_regions(impl_source);
    if impl_source.has_non_region_infer() {
        // Unused generic types or consts on an impl get replaced with inference vars,
        // but never resolved, causing the return value of a query to contain inference
        // vars. We do not have a concept for this and will in fact ICE in stable hashing
        // of the return value. So bail out instead.
        match impl_source {
            ImplSource::UserDefined(impl_) => {
                tcx.dcx().span_delayed_bug(
                    tcx.def_span(impl_.impl_def_id),
                    "This implementation has unconstrained generic parameters. Consider reviewing the definition."
                );
            }
            _ => unreachable!("Unexpected ImplSource variant with unresolved inference variables."),
        }
        return Err(CodegenObligationError::FulfillmentError);
    }

    Ok(&*tcx.arena.alloc(impl_source))
}
