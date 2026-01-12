//! This module is very similar to `compare_impl_item`.
//! Most logic is taken from there,
//! since in a very similar way we're comparing some declaration of a signature to an implementation.
//! The major difference is that we don't bother with self types, since for EIIs we're comparing freestanding item.

use std::borrow::Cow;
use std::iter;

use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::{Applicability, E0806, struct_span_code_err};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{self as hir, FnSig, HirId, ItemKind};
use rustc_infer::infer::{self, InferCtxt, TyCtxtInferExt};
use rustc_infer::traits::{ObligationCause, ObligationCauseCode};
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt, TypingMode};
use rustc_span::{ErrorGuaranteed, Ident, Span, Symbol};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::regions::InferCtxtRegionExt;
use rustc_trait_selection::traits::{self, ObligationCtxt};
use tracing::{debug, instrument};

use super::potentially_plural_count;
use crate::check::compare_impl_item::{
    CheckNumberOfEarlyBoundRegionsError, check_number_of_early_bound_regions,
};
use crate::errors::{EiiWithGenerics, LifetimesOrBoundsMismatchOnEii};

/// Checks whether the signature of some `external_impl`, matches
/// the signature of `declaration`, which it is supposed to be compatible
/// with in order to implement the item.
pub(crate) fn compare_eii_function_types<'tcx>(
    tcx: TyCtxt<'tcx>,
    external_impl: LocalDefId,
    foreign_item: DefId,
    eii_name: Symbol,
    eii_attr_span: Span,
) -> Result<(), ErrorGuaranteed> {
    check_is_structurally_compatible(tcx, external_impl, foreign_item, eii_name, eii_attr_span)?;

    let external_impl_span = tcx.def_span(external_impl);
    let cause = ObligationCause::new(
        external_impl_span,
        external_impl,
        ObligationCauseCode::CompareEii { external_impl, declaration: foreign_item },
    );

    // FIXME(eii): even if we don't support generic functions, we should support explicit outlive bounds here
    let param_env = tcx.param_env(foreign_item);

    let infcx = &tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let ocx = ObligationCtxt::new_with_diagnostics(infcx);

    // We now need to check that the signature of the implementation is
    // compatible with that of the declaration. We do this by
    // checking that `impl_fty <: trait_fty`.
    //
    // FIXME: We manually instantiate the declaration here as we need
    // to manually compute its implied bounds. Otherwise this could just
    // be ocx.sub(impl_sig, trait_sig).

    let mut wf_tys = FxIndexSet::default();
    let norm_cause = ObligationCause::misc(external_impl_span, external_impl);

    let declaration_sig = tcx.fn_sig(foreign_item).instantiate_identity();
    let declaration_sig = tcx.liberate_late_bound_regions(external_impl.into(), declaration_sig);
    debug!(?declaration_sig);

    let unnormalized_external_impl_sig = infcx.instantiate_binder_with_fresh_vars(
        external_impl_span,
        infer::BoundRegionConversionTime::HigherRankedType,
        tcx.fn_sig(external_impl).instantiate(
            tcx,
            infcx.fresh_args_for_item(external_impl_span, external_impl.to_def_id()),
        ),
    );
    let external_impl_sig = ocx.normalize(&norm_cause, param_env, unnormalized_external_impl_sig);
    debug!(?external_impl_sig);

    // Next, add all inputs and output as well-formed tys. Importantly,
    // we have to do this before normalization, since the normalized ty may
    // not contain the input parameters. See issue #87748.
    wf_tys.extend(declaration_sig.inputs_and_output.iter());
    let declaration_sig = ocx.normalize(&norm_cause, param_env, declaration_sig);
    // We also have to add the normalized declaration
    // as we don't normalize during implied bounds computation.
    wf_tys.extend(external_impl_sig.inputs_and_output.iter());

    // FIXME: Copied over from compare impl items, same issue:
    // We'd want to keep more accurate spans than "the method signature" when
    // processing the comparison between the trait and impl fn, but we sadly lose them
    // and point at the whole signature when a trait bound or specific input or output
    // type would be more appropriate. In other places we have a `Vec<Span>`
    // corresponding to their `Vec<Predicate>`, but we don't have that here.
    // Fixing this would improve the output of test `issue-83765.rs`.
    let result = ocx.sup(&cause, param_env, declaration_sig, external_impl_sig);

    if let Err(terr) = result {
        debug!(?external_impl_sig, ?declaration_sig, ?terr, "sub_types failed");

        let emitted = report_eii_mismatch(
            infcx,
            cause,
            param_env,
            terr,
            (foreign_item, declaration_sig),
            (external_impl, external_impl_sig),
            eii_attr_span,
            eii_name,
        );
        return Err(emitted);
    }

    if !(declaration_sig, external_impl_sig).references_error() {
        for ty in unnormalized_external_impl_sig.inputs_and_output {
            ocx.register_obligation(traits::Obligation::new(
                infcx.tcx,
                cause.clone(),
                param_env,
                ty::ClauseKind::WellFormed(ty.into()),
            ));
        }
    }

    // Check that all obligations are satisfied by the implementation's
    // version.
    let errors = ocx.evaluate_obligations_error_on_ambiguity();
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

/// Checks a bunch of different properties of the impl/trait methods for
/// compatibility, such as asyncness, number of argument, self receiver kind,
/// and number of early- and late-bound generics.
///
/// Corresponds to `check_method_is_structurally_compatible` for impl method compatibility checks.
fn check_is_structurally_compatible<'tcx>(
    tcx: TyCtxt<'tcx>,
    external_impl: LocalDefId,
    declaration: DefId,
    eii_name: Symbol,
    eii_attr_span: Span,
) -> Result<(), ErrorGuaranteed> {
    check_no_generics(tcx, external_impl, declaration, eii_name, eii_attr_span)?;
    check_number_of_arguments(tcx, external_impl, declaration, eii_name, eii_attr_span)?;
    check_early_region_bounds(tcx, external_impl, declaration, eii_attr_span)?;
    Ok(())
}

/// externally implementable items can't have generics
fn check_no_generics<'tcx>(
    tcx: TyCtxt<'tcx>,
    external_impl: LocalDefId,
    _declaration: DefId,
    eii_name: Symbol,
    eii_attr_span: Span,
) -> Result<(), ErrorGuaranteed> {
    let generics = tcx.generics_of(external_impl);
    if generics.own_requires_monomorphization() {
        tcx.dcx().emit_err(EiiWithGenerics {
            span: tcx.def_span(external_impl),
            attr: eii_attr_span,
            eii_name,
        });
    }

    Ok(())
}

fn check_early_region_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    external_impl: LocalDefId,
    declaration: DefId,
    eii_attr_span: Span,
) -> Result<(), ErrorGuaranteed> {
    let external_impl_generics = tcx.generics_of(external_impl.to_def_id());
    let external_impl_params = external_impl_generics.own_counts().lifetimes;

    let declaration_generics = tcx.generics_of(declaration);
    let declaration_params = declaration_generics.own_counts().lifetimes;

    let Err(CheckNumberOfEarlyBoundRegionsError { span, generics_span, bounds_span, where_span }) =
        check_number_of_early_bound_regions(
            tcx,
            external_impl,
            declaration,
            external_impl_generics,
            external_impl_params,
            declaration_generics,
            declaration_params,
        )
    else {
        return Ok(());
    };

    let mut diag = tcx.dcx().create_err(LifetimesOrBoundsMismatchOnEii {
        span,
        ident: tcx.item_name(external_impl.to_def_id()),
        generics_span,
        bounds_span,
        where_span,
    });

    diag.span_label(eii_attr_span, format!("required because of this attribute"));
    return Err(diag.emit());
}

fn check_number_of_arguments<'tcx>(
    tcx: TyCtxt<'tcx>,
    external_impl: LocalDefId,
    declaration: DefId,
    eii_name: Symbol,
    eii_attr_span: Span,
) -> Result<(), ErrorGuaranteed> {
    let external_impl_fty = tcx.fn_sig(external_impl);
    let declaration_fty = tcx.fn_sig(declaration);
    let declaration_number_args = declaration_fty.skip_binder().inputs().skip_binder().len();
    let external_impl_number_args = external_impl_fty.skip_binder().inputs().skip_binder().len();

    // if the number of args are equal, we're trivially done
    if declaration_number_args == external_impl_number_args {
        Ok(())
    } else {
        Err(report_number_of_arguments_mismatch(
            tcx,
            external_impl,
            declaration,
            eii_name,
            eii_attr_span,
            declaration_number_args,
            external_impl_number_args,
        ))
    }
}

fn report_number_of_arguments_mismatch<'tcx>(
    tcx: TyCtxt<'tcx>,
    external_impl: LocalDefId,
    declaration: DefId,
    eii_name: Symbol,
    eii_attr_span: Span,
    declaration_number_args: usize,
    external_impl_number_args: usize,
) -> ErrorGuaranteed {
    let external_impl_name = tcx.item_name(external_impl.to_def_id());

    let declaration_span = declaration
        .as_local()
        .and_then(|def_id| {
            let declaration_sig = get_declaration_sig(tcx, def_id).expect("foreign item sig");
            let pos = declaration_number_args.saturating_sub(1);
            declaration_sig.decl.inputs.get(pos).map(|arg| {
                if pos == 0 {
                    arg.span
                } else {
                    arg.span.with_lo(declaration_sig.decl.inputs[0].span.lo())
                }
            })
        })
        .or_else(|| tcx.hir_span_if_local(declaration))
        .unwrap_or_else(|| tcx.def_span(declaration));

    let (_, external_impl_sig, _, _) = &tcx.hir_expect_item(external_impl).expect_fn();
    let pos = external_impl_number_args.saturating_sub(1);
    let impl_span = external_impl_sig
        .decl
        .inputs
        .get(pos)
        .map(|arg| {
            if pos == 0 {
                arg.span
            } else {
                arg.span.with_lo(external_impl_sig.decl.inputs[0].span.lo())
            }
        })
        .unwrap_or_else(|| tcx.def_span(external_impl));

    let mut err = struct_span_code_err!(
        tcx.dcx(),
        impl_span,
        E0806,
        "`{external_impl_name}` has {} but #[{eii_name}] requires it to have {}",
        potentially_plural_count(external_impl_number_args, "parameter"),
        declaration_number_args
    );

    err.span_label(
        declaration_span,
        format!("requires {}", potentially_plural_count(declaration_number_args, "parameter")),
    );

    err.span_label(
        impl_span,
        format!(
            "expected {}, found {}",
            potentially_plural_count(declaration_number_args, "parameter"),
            external_impl_number_args
        ),
    );

    err.span_label(eii_attr_span, format!("required because of this attribute"));

    err.emit()
}

fn report_eii_mismatch<'tcx>(
    infcx: &InferCtxt<'tcx>,
    mut cause: ObligationCause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    terr: TypeError<'tcx>,
    (declaration_did, declaration_sig): (DefId, ty::FnSig<'tcx>),
    (external_impl_did, external_impl_sig): (LocalDefId, ty::FnSig<'tcx>),
    eii_attr_span: Span,
    eii_name: Symbol,
) -> ErrorGuaranteed {
    let tcx = infcx.tcx;
    let (impl_err_span, trait_err_span, external_impl_name) =
        extract_spans_for_error_reporting(infcx, terr, &cause, declaration_did, external_impl_did);

    let mut diag = struct_span_code_err!(
        tcx.dcx(),
        impl_err_span,
        E0806,
        "function `{}` has a type that is incompatible with the declaration of `#[{eii_name}]`",
        external_impl_name
    );

    diag.span_note(eii_attr_span, "expected this because of this attribute");

    match &terr {
        TypeError::ArgumentMutability(i) | TypeError::ArgumentSorts(_, i) => {
            if declaration_sig.inputs().len() == *i {
                // Suggestion to change output type. We do not suggest in `async` functions
                // to avoid complex logic or incorrect output.
                if let ItemKind::Fn { sig, .. } = &tcx.hir_expect_item(external_impl_did).kind
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
        let item = tcx.hir_expect_item(external_impl);
        let (ident, sig, _, _) = item.expect_fn();
        (sig.decl.inputs.iter().map(|t| t.span).chain(iter::once(sig.decl.output.span())), ident)
    };

    let declaration_args = declaration.as_local().map(|def_id| {
        if let Some(sig) = get_declaration_sig(tcx, def_id) {
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
        _ => (
            cause.span,
            tcx.hir_span_if_local(declaration).or_else(|| Some(tcx.def_span(declaration))),
            external_impl_name,
        ),
    }
}

fn get_declaration_sig<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> Option<&'tcx FnSig<'tcx>> {
    let hir_id: HirId = tcx.local_def_id_to_hir_id(def_id);
    tcx.hir_fn_sig_by_hir_id(hir_id)
}
