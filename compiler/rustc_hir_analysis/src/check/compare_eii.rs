use std::borrow::Cow;
use std::iter;

use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::{Applicability, E0805, struct_span_code_err};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{self as hir, FnSig, HirId, ItemKind};
use rustc_infer::infer::{self, InferCtxt, TyCtxtInferExt};
use rustc_infer::traits::{ObligationCause, ObligationCauseCode};
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::{self, TyCtxt, TypingMode};
use rustc_span::{ErrorGuaranteed, Ident, Span, Symbol};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::regions::InferCtxtRegionExt;
use rustc_trait_selection::traits::ObligationCtxt;
use tracing::{debug, instrument};

use super::potentially_plural_count;
use crate::errors::{EiiWithGenerics, LifetimesOrBoundsMismatchOnEII};

/// Checks a bunch of different properties of the impl/trait methods for
/// compatibility, such as asyncness, number of argument, self receiver kind,
/// and number of early- and late-bound generics.
fn check_is_structurally_compatible<'tcx>(
    tcx: TyCtxt<'tcx>,
    external_impl: LocalDefId,
    declaration: DefId,
    eii_name: Symbol,
    eii_attr_span: Span,
) -> Result<(), ErrorGuaranteed> {
    check_no_generics(tcx, external_impl, declaration, eii_name, eii_attr_span)?;
    compare_number_of_method_arguments(tcx, external_impl, declaration, eii_name, eii_attr_span)?;
    check_region_bounds_on_impl_item(tcx, external_impl, declaration, eii_attr_span)?;
    Ok(())
}

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

fn check_region_bounds_on_impl_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    external_impl: LocalDefId,
    declaration: DefId,
    eii_attr_span: Span,
) -> Result<(), ErrorGuaranteed> {
    let external_impl_generics = tcx.generics_of(external_impl.to_def_id());
    let external_impl_params = external_impl_generics.own_counts().lifetimes;

    let declaration_generics = tcx.generics_of(declaration);
    let declaration_params = declaration_generics.own_counts().lifetimes;

    debug!(?declaration_generics, ?external_impl_generics);

    // Must have same number of early-bound lifetime parameters.
    // Unfortunately, if the user screws up the bounds, then this
    // will change classification between early and late. E.g.,
    // if in trait we have `<'a,'b:'a>`, and in impl we just have
    // `<'a,'b>`, then we have 2 early-bound lifetime parameters
    // in trait but 0 in the impl. But if we report "expected 2
    // but found 0" it's confusing, because it looks like there
    // are zero. Since I don't quite know how to phrase things at
    // the moment, give a kind of vague error message.
    if declaration_params != external_impl_params {
        let span = tcx
            .hir_get_generics(external_impl)
            .expect("expected impl item to have generics or else we can't compare them")
            .span;

        let mut generics_span = None;
        let mut bounds_span = vec![];
        let mut where_span = None;

        if let Some(declaration_node) = tcx.hir_get_if_local(declaration)
            && let Some(declaration_generics) = declaration_node.generics()
        {
            generics_span = Some(declaration_generics.span);
            // FIXME: we could potentially look at the impl's bounds to not point at bounds that
            // *are* present in the impl.
            for p in declaration_generics.predicates {
                if let hir::WherePredicateKind::BoundPredicate(pred) = p.kind {
                    for b in pred.bounds {
                        if let hir::GenericBound::Outlives(lt) = b {
                            bounds_span.push(lt.ident.span);
                        }
                    }
                }
            }
            if let Some(implementation_generics) = tcx.hir_get_generics(external_impl) {
                let mut impl_bounds = 0;
                for p in implementation_generics.predicates {
                    if let hir::WherePredicateKind::BoundPredicate(pred) = p.kind {
                        for b in pred.bounds {
                            if let hir::GenericBound::Outlives(_) = b {
                                impl_bounds += 1;
                            }
                        }
                    }
                }
                if impl_bounds == bounds_span.len() {
                    bounds_span = vec![];
                } else if implementation_generics.has_where_clause_predicates {
                    where_span = Some(implementation_generics.where_clause_span);
                }
            }
        }
        let mut diag = tcx.dcx().create_err(LifetimesOrBoundsMismatchOnEII {
            span,
            ident: tcx.item_name(external_impl.to_def_id()),
            generics_span,
            bounds_span,
            where_span,
        });

        diag.span_label(eii_attr_span, format!("required because of this attribute"));
        return Err(diag.emit());
    }

    Ok(())
}

fn compare_number_of_method_arguments<'tcx>(
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
    let external_impl_name = tcx.item_name(external_impl.to_def_id());

    if declaration_number_args != external_impl_number_args {
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
            E0805,
            "`{external_impl_name}` has {} but #[{eii_name}] requires it to have {}",
            potentially_plural_count(external_impl_number_args, "parameter"),
            declaration_number_args
        );

        // if let Some(declaration_span) = declaration_span {
        err.span_label(
            declaration_span,
            format!("requires {}", potentially_plural_count(declaration_number_args, "parameter")),
        );
        // }

        err.span_label(
            impl_span,
            format!(
                "expected {}, found {}",
                potentially_plural_count(declaration_number_args, "parameter"),
                external_impl_number_args
            ),
        );

        err.span_label(eii_attr_span, format!("required because of this attribute"));

        return Err(err.emit());
    }

    Ok(())
}

// checks whether the signature of some `external_impl`, matches
// the signature of `declaration`, which it is supposed to be compatible
// with in order to implement the item.
pub(crate) fn compare_eii_function_types<'tcx>(
    tcx: TyCtxt<'tcx>,
    external_impl: LocalDefId,
    declaration: DefId,
    eii_name: Symbol,
    eii_attr_span: Span,
) -> Result<(), ErrorGuaranteed> {
    check_is_structurally_compatible(tcx, external_impl, declaration, eii_name, eii_attr_span)?;

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
    let norm_cause = ObligationCause::misc(external_impl_span, external_impl);

    let declaration_sig = tcx.fn_sig(declaration).instantiate_identity();
    let declaration_sig = infcx.enter_forall_and_leak_universe(declaration_sig);
    let declaration_sig = ocx.normalize(&norm_cause, param_env, declaration_sig);

    let external_impl_sig = infcx.instantiate_binder_with_fresh_vars(
        external_impl_span,
        infer::HigherRankedType,
        tcx.fn_sig(external_impl).instantiate(
            tcx,
            infcx.fresh_args_for_item(external_impl_span, external_impl.to_def_id()),
        ),
    );
    let external_impl_sig = ocx.normalize(&norm_cause, param_env, external_impl_sig);
    debug!(?external_impl_sig);

    // FIXME: We'd want to keep more accurate spans than "the method signature" when
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
            (declaration, declaration_sig),
            (external_impl, external_impl_sig),
            eii_attr_span,
            eii_name,
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
    eii_attr_span: Span,
    eii_name: Symbol,
) -> ErrorGuaranteed {
    let tcx = infcx.tcx;
    let (impl_err_span, trait_err_span, external_impl_name) =
        extract_spans_for_error_reporting(infcx, terr, &cause, declaration_did, external_impl_did);

    let mut diag = struct_span_code_err!(
        tcx.dcx(),
        impl_err_span,
        E0805,
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
