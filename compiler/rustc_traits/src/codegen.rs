// This file contains various trait resolution methods used by codegen.
// They all assume regions can be erased and monomorphic types. It
// seems likely that they should eventually be merged into more
// general routines.

use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::bug;
use rustc_middle::traits::CodegenObligationError;
use rustc_middle::ty::{self, PseudoCanonicalInput, TyCtxt, TypeVisitableExt};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::solve::InferCtxtSelectExt as _;
use rustc_trait_selection::traits::{
    ImplSource, Obligation, ObligationCause, ObligationCtxt, ScrubbedTraitError, SelectionContext,
    SelectionError,
};
use tracing::debug;

/// Attempts to resolve an obligation to an `ImplSource`. The result is
/// a shallow `ImplSource` resolution, meaning that we do not
/// (necessarily) resolve all nested obligations on the impl. Note
/// that type check should guarantee to us that all nested
/// obligations *could be* resolved if we wanted to.
///
/// This also expects that `trait_ref` is fully normalized.
///
/// When `use_const` is set, we assume the new trait solver is enabled and use
/// `HostEffectPredicate` with `constness` set to `Const`.
pub(crate) fn codegen_select_candidate_inner<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    use_const: bool,
) -> Result<&'tcx ImplSource<'tcx, ()>, CodegenObligationError> {
    // We expect the input to be fully normalized.
    tcx.debug_assert_fully_normalized(typing_env, trait_ref);

    // Do the initial selection for the obligation. This yields the
    // shallow result we are looking for -- that is, what specific impl.
    let (infcx, param_env) = tcx.infer_ctxt().ignoring_regions().build_with_typing_env(typing_env);
    let mut selcx = SelectionContext::new(&infcx);

    let obligation_cause = ObligationCause::dummy();

    let selection = if use_const {
        let host_predicate =
            ty::HostEffectPredicate { trait_ref, constness: ty::BoundConstness::Const };
        let obligation = Obligation::new(tcx, obligation_cause, param_env, host_predicate);

        infcx.select_host_effect_predicate_in_new_trait_solver(&obligation)
    } else {
        let obligation = Obligation::new(tcx, obligation_cause, param_env, trait_ref);
        selcx.select(&obligation)
    };

    let selection = match selection {
        Ok(Some(selection)) => selection,
        Ok(None) => return Err(CodegenObligationError::Ambiguity),
        Err(SelectionError::Unimplemented) => return Err(CodegenObligationError::Unimplemented),
        Err(e) => {
            bug!(
                "Encountered error `{e:?}` selecting `{trait_ref:?}` during codegen (use_const: {use_const})"
            )
        }
    };

    debug!(?selection);

    // Currently, we use a fulfillment context to completely resolve
    // all nested obligations. This is because they can inform the
    // inference of the impl's type parameters.
    let ocx = ObligationCtxt::new(&infcx);
    let impl_source = selection.map(|obligation| {
        ocx.register_obligation(obligation);
    });

    // In principle, we only need to do this so long as `impl_source`
    // contains unbound type parameters. It could be a slight
    // optimization to stop iterating early.
    let errors = ocx.evaluate_obligations_error_on_ambiguity();
    if !errors.is_empty() {
        // `rustc_monomorphize::collector` assumes there are no type errors.
        // Cycle errors are the only post-monomorphization errors possible; emit them now so
        // `rustc_ty_utils::resolve_associated_item` doesn't return `None` post-monomorphization.
        for err in errors {
            if let ScrubbedTraitError::Cycle(cycle) = err {
                infcx.err_ctxt().report_overflow_obligation_cycle(&cycle);
            }
        }
        return Err(CodegenObligationError::Unimplemented);
    }

    let impl_source = infcx.resolve_vars_if_possible(impl_source);
    let impl_source = tcx.erase_and_anonymize_regions(impl_source);
    if impl_source.has_non_region_infer() {
        // Unused generic types or consts on an impl get replaced with inference vars,
        // but never resolved, causing the return value of a query to contain inference
        // vars. We do not have a concept for this and will in fact ICE in stable hashing
        // of the return value. So bail out instead.
        let guar = match impl_source {
            ImplSource::UserDefined(impl_) => tcx.dcx().span_delayed_bug(
                tcx.def_span(impl_.impl_def_id),
                "this impl has unconstrained generic parameters",
            ),
            _ => unreachable!(),
        };
        return Err(CodegenObligationError::UnconstrainedParam(guar));
    }

    Ok(&*tcx.arena.alloc(impl_source))
}

pub(crate) fn codegen_select_candidate<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: PseudoCanonicalInput<'tcx, ty::TraitRef<'tcx>>,
) -> Result<&'tcx ImplSource<'tcx, ()>, CodegenObligationError> {
    codegen_select_candidate_inner(tcx, key.typing_env, key.value, false)
}

pub(crate) fn codegen_select_candidate_for_ctfe<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: PseudoCanonicalInput<'tcx, ty::TraitRef<'tcx>>,
) -> Result<&'tcx ImplSource<'tcx, ()>, CodegenObligationError> {
    codegen_select_candidate_inner(tcx, key.typing_env, key.value, true)
}
