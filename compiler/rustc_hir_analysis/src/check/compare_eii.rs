use std::borrow::Cow;
use std::iter;

use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::{Applicability, E0053, E0053, struct_span_code_err, struct_span_code_err};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{self as hir, self as hir, HirId, HirId, ItemKind, ItemKind};
use rustc_infer::infer::{self, InferCtxt, TyCtxtInferExt};
use rustc_infer::traits::{ObligationCause, ObligationCauseCode};
use rustc_middle::ty;
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::error::{ExpectedFound, TypeError, TypeError};
use rustc_span::{ErrorGuaranteed, Ident, Span};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::regions::InferCtxtRegionExt;
use rustc_trait_selection::traits::ObligationCtxt;
use rustc_type_ir::TypingMode;
use tracing::{debug, instrument};

// checks whether the signature of some `external_impl`, matches
// the signature of `declaration`, which it is supposed to be compatible
// with in order to implement the item.
pub(crate) fn compare_eii_function_types<'tcx>(
    tcx: TyCtxt<'tcx>,
    external_impl: LocalDefId,
    declaration: DefId,
) -> Result<(), ErrorGuaranteed> {
    let external_impl_span = tcx.def_span(external_impl);
    let cause = ObligationCause::new(
        external_impl_span,
        external_impl,
        ObligationCauseCode::CompareEII { external_impl, declaration },
    );

    // no trait bounds
    let param_env = ty::ParamEnv::empty();

    let infcx = &tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let ocx = ObligationCtxt::new_with_diagnostics(infcx);

    // We now need to check that the signature of the impl method is
    // compatible with that of the trait method. We do this by
    // checking that `impl_fty <: trait_fty`.
    //
    // FIXME. Unfortunately, this doesn't quite work right now because
    // associated type normalization is not integrated into subtype
    // checks. For the comparison to be valid, we need to
    // normalize the associated types in the impl/trait methods
    // first. However, because function types bind regions, just
    // calling `FnCtxt::normalize` would have no effect on
    // any associated types appearing in the fn arguments or return
    // type.

    let wf_tys = FxIndexSet::default();
    let external_impl_sig = infcx.instantiate_binder_with_fresh_vars(
        external_impl_span,
        infer::HigherRankedType,
        tcx.fn_sig(external_impl).instantiate(
            tcx,
            infcx.fresh_args_for_item(external_impl_span, external_impl.to_def_id()),
        ),
    );

    let norm_cause = ObligationCause::misc(external_impl_span, external_impl);
    let external_impl_sig = ocx.normalize(&norm_cause, param_env, external_impl_sig);
    debug!(?external_impl_sig);

    let declaration_sig = tcx.fn_sig(declaration).instantiate_identity();
    let declaration_sig =
        tcx.liberate_late_bound_regions(external_impl.to_def_id(), declaration_sig);
    let declaration_sig = ocx.normalize(&norm_cause, param_env, declaration_sig);

    // FIXME: We'd want to keep more accurate spans than "the method signature" when
    // processing the comparison between the trait and impl fn, but we sadly lose them
    // and point at the whole signature when a trait bound or specific input or output
    // type would be more appropriate. In other places we have a `Vec<Span>`
    // corresponding to their `Vec<Predicate>`, but we don't have that here.
    // Fixing this would improve the output of test `issue-83765.rs`.
    let result = ocx.sup(&cause, param_env, declaration_sig, external_impl_sig);

    if let Err(terr) = result {
        debug!(?external_impl_sig, ?declaration_sig, ?terr, "sub_types failed");

        // TODO: nice error
        let emitted = report_eii_mismatch(
            infcx,
            cause,
            param_env,
            terr,
            (declaration, declaration_sig),
            (external_impl, external_impl_sig),
        );
        return Err(emitted);
    }

    // Check that all obligations are satisfied by the implementation's
    // version.
    let errors = ocx.select_all_or_error();
    if !errors.is_empty() {
        let reported = infcx.err_ctxt().report_fulfillment_errors(errors);
        return Err(reported);
    }

    // Finally, resolve all regions. This catches wily misuses of
    // lifetime parameters.
    let errors = infcx.resolve_regions(external_impl, param_env, wf_tys);
    if !errors.is_empty() {
        return Err(infcx
            .tainted_by_errors()
            .unwrap_or_else(|| infcx.err_ctxt().report_region_errors(external_impl, &errors)));
    }

    Ok(())
}

fn report_eii_mismatch<'tcx>(
    infcx: &InferCtxt<'tcx>,
    mut cause: ObligationCause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    terr: TypeError<'tcx>,
    (declaration_did, declaration_sig): (DefId, ty::FnSig<'tcx>),
    (external_impl_did, external_impl_sig): (LocalDefId, ty::FnSig<'tcx>),
) -> ErrorGuaranteed {
    let tcx = infcx.tcx;
    let (impl_err_span, trait_err_span, external_impl_name) =
        extract_spans_for_error_reporting(infcx, terr, &cause, declaration_did, external_impl_did);

    let mut diag = struct_span_code_err!(
        tcx.dcx(),
        impl_err_span,
        E0053, // TODO: new error code
        "function `{}` has a type that is incompatible with the declaration",
        external_impl_name
    );
    match &terr {
        TypeError::ArgumentMutability(i) | TypeError::ArgumentSorts(_, i) => {
            if declaration_sig.inputs().len() == *i {
                // Suggestion to change output type. We do not suggest in `async` functions
                // to avoid complex logic or incorrect output.
                if let ItemKind::Fn { sig, .. } = &tcx.hir().expect_item(external_impl_did).kind
                    && !sig.header.asyncness.is_async()
                {
                    let msg = "change the output type to match the declaration";
                    let ap = Applicability::MachineApplicable;
                    match sig.decl.output {
                        hir::FnRetTy::DefaultReturn(sp) => {
                            let sugg = format!(" -> {}", declaration_sig.output());
                            diag.span_suggestion_verbose(sp, msg, sugg, ap);
                        }
                        hir::FnRetTy::Return(hir_ty) => {
                            let sugg = declaration_sig.output();
                            diag.span_suggestion_verbose(hir_ty.span, msg, sugg, ap);
                        }
                    };
                };
            } else if let Some(trait_ty) = declaration_sig.inputs().get(*i) {
                diag.span_suggestion_verbose(
                    impl_err_span,
                    "change the parameter type to match the declaration",
                    trait_ty,
                    Applicability::MachineApplicable,
                );
            }
        }
        _ => {}
    }

    cause.span = impl_err_span;
    infcx.err_ctxt().note_type_err(
        &mut diag,
        &cause,
        trait_err_span.map(|sp| (sp, Cow::from("type in declaration"), false)),
        Some(param_env.and(infer::ValuePairs::PolySigs(ExpectedFound {
            expected: ty::Binder::dummy(declaration_sig),
            found: ty::Binder::dummy(external_impl_sig),
        }))),
        terr,
        false,
        None,
    );

    diag.emit()
}

#[instrument(level = "debug", skip(infcx))]
fn extract_spans_for_error_reporting<'tcx>(
    infcx: &infer::InferCtxt<'tcx>,
    terr: TypeError<'_>,
    cause: &ObligationCause<'tcx>,
    declaration: DefId,
    external_impl: LocalDefId,
) -> (Span, Option<Span>, Ident) {
    let tcx = infcx.tcx;
    let (mut external_impl_args, external_impl_name) = {
        let item = tcx.hir().expect_item(external_impl);
        let (sig, _, _) = item.expect_fn();
        (
            sig.decl.inputs.iter().map(|t| t.span).chain(iter::once(sig.decl.output.span())),
            item.ident,
        )
    };

    let declaration_args = declaration.as_local().map(|def_id| {
        let hir_id: HirId = tcx.local_def_id_to_hir_id(def_id);
        if let Some(sig) = tcx.hir_fn_sig_by_hir_id(hir_id) {
            sig.decl.inputs.iter().map(|t| t.span).chain(iter::once(sig.decl.output.span()))
        } else {
            panic!("expected {def_id:?} to be a foreign function");
        }
    });

    match terr {
        TypeError::ArgumentMutability(i) | TypeError::ArgumentSorts(ExpectedFound { .. }, i) => (
            external_impl_args.nth(i).unwrap(),
            declaration_args.and_then(|mut args| args.nth(i)),
            external_impl_name,
        ),
        _ => (cause.span, tcx.hir().span_if_local(declaration), external_impl_name),
    }
}
