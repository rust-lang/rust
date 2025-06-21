use std::ops::Not;

use rustc_abi::ExternAbi;
use rustc_attr_data_structures::{AttributeKind, find_attr};
use rustc_hir as hir;
use rustc_hir::Node;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::span_bug;
use rustc_middle::ty::{self, TyCtxt, TypingMode};
use rustc_session::config::EntryFnType;
use rustc_span::Span;
use rustc_span::def_id::{CRATE_DEF_ID, DefId, LocalDefId};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::traits::{self, ObligationCause, ObligationCauseCode};

use super::check_function_signature;
use crate::errors;

pub(crate) fn check_for_entry_fn(tcx: TyCtxt<'_>) {
    match tcx.entry_fn(()) {
        Some((def_id, EntryFnType::Main { .. })) => check_main_fn_ty(tcx, def_id),
        _ => {}
    }
}

fn check_main_fn_ty(tcx: TyCtxt<'_>, main_def_id: DefId) {
    let main_fnsig = tcx.fn_sig(main_def_id).instantiate_identity();
    let main_span = tcx.def_span(main_def_id);

    fn main_fn_diagnostics_def_id(tcx: TyCtxt<'_>, def_id: DefId, sp: Span) -> LocalDefId {
        if let Some(local_def_id) = def_id.as_local() {
            let hir_type = tcx.type_of(local_def_id).instantiate_identity();
            if !matches!(hir_type.kind(), ty::FnDef(..)) {
                span_bug!(sp, "main has a non-function type: found `{}`", hir_type);
            }
            local_def_id
        } else {
            CRATE_DEF_ID
        }
    }

    fn main_fn_generics_params_span(tcx: TyCtxt<'_>, def_id: DefId) -> Option<Span> {
        if !def_id.is_local() {
            return None;
        }
        match tcx.hir_node_by_def_id(def_id.expect_local()) {
            Node::Item(hir::Item { kind: hir::ItemKind::Fn { generics, .. }, .. }) => {
                generics.params.is_empty().not().then_some(generics.span)
            }
            _ => {
                span_bug!(tcx.def_span(def_id), "main has a non-function type");
            }
        }
    }

    fn main_fn_where_clauses_span(tcx: TyCtxt<'_>, def_id: DefId) -> Option<Span> {
        if !def_id.is_local() {
            return None;
        }
        match tcx.hir_node_by_def_id(def_id.expect_local()) {
            Node::Item(hir::Item { kind: hir::ItemKind::Fn { generics, .. }, .. }) => {
                Some(generics.where_clause_span)
            }
            _ => {
                span_bug!(tcx.def_span(def_id), "main has a non-function type");
            }
        }
    }

    fn main_fn_asyncness_span(tcx: TyCtxt<'_>, def_id: DefId) -> Option<Span> {
        if !def_id.is_local() {
            return None;
        }
        Some(tcx.def_span(def_id))
    }

    fn main_fn_return_type_span(tcx: TyCtxt<'_>, def_id: DefId) -> Option<Span> {
        if !def_id.is_local() {
            return None;
        }
        match tcx.hir_node_by_def_id(def_id.expect_local()) {
            Node::Item(hir::Item { kind: hir::ItemKind::Fn { sig: fn_sig, .. }, .. }) => {
                Some(fn_sig.decl.output.span())
            }
            _ => {
                span_bug!(tcx.def_span(def_id), "main has a non-function type");
            }
        }
    }

    let mut error = false;
    let main_diagnostics_def_id = main_fn_diagnostics_def_id(tcx, main_def_id, main_span);

    let main_asyncness = tcx.asyncness(main_def_id);
    if main_asyncness.is_async() {
        let asyncness_span = main_fn_asyncness_span(tcx, main_def_id);
        tcx.dcx()
            .emit_err(errors::MainFunctionAsync { span: main_span, asyncness: asyncness_span });
        error = true;
    }

    if let Some(attr_span) =
        find_attr!(tcx.get_all_attrs(main_def_id), AttributeKind::TrackCaller(span) => *span)
    {
        tcx.dcx().emit_err(errors::TrackCallerOnMain { span: attr_span, annotated: main_span });
        error = true;
    }

    if !tcx.codegen_fn_attrs(main_def_id).target_features.is_empty()
        // Calling functions with `#[target_feature]` is not unsafe on WASM, see #84988
        && !tcx.sess.target.is_like_wasm
        && !tcx.sess.opts.actually_rustdoc
    {
        tcx.dcx().emit_err(errors::TargetFeatureOnMain { main: main_span });
        error = true;
    }

    if error {
        return;
    }

    // Main should have no WC, so empty param env is OK here.
    let param_env = ty::ParamEnv::empty();
    let expected_return_type;
    if let Some(term_did) = tcx.lang_items().termination() {
        let return_ty = main_fnsig.output();
        let return_ty_span = main_fn_return_type_span(tcx, main_def_id).unwrap_or(main_span);
        let Some(return_ty) = return_ty.no_bound_vars() else {
            tcx.dcx().emit_err(errors::MainFunctionReturnTypeGeneric { span: return_ty_span });
            return;
        };
        let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
        let cause = traits::ObligationCause::new(
            return_ty_span,
            main_diagnostics_def_id,
            ObligationCauseCode::MainFunctionType,
        );
        let ocx = traits::ObligationCtxt::new_with_diagnostics(&infcx);
        let norm_return_ty = ocx.normalize(&cause, param_env, return_ty);
        ocx.register_bound(cause, param_env, norm_return_ty, term_did);
        let errors = ocx.select_all_or_error();
        if !errors.is_empty() {
            infcx.err_ctxt().report_fulfillment_errors(errors);
            error = true;
        }
        // now we can take the return type of the given main function
        expected_return_type = norm_return_ty;
    } else {
        // standard () main return type
        expected_return_type = tcx.types.unit;
    }

    if error {
        return;
    }

    let expected_sig = ty::Binder::dummy(tcx.mk_fn_sig(
        [],
        expected_return_type,
        false,
        hir::Safety::Safe,
        ExternAbi::Rust,
    ));

    if check_function_signature(
        tcx,
        ObligationCause::new(
            main_span,
            main_diagnostics_def_id,
            ObligationCauseCode::MainFunctionType,
        ),
        main_def_id,
        expected_sig,
    )
    .is_err()
    {
        return;
    }

    let main_fn_generics = tcx.generics_of(main_def_id);
    let main_fn_predicates = tcx.predicates_of(main_def_id);
    if main_fn_generics.count() != 0 || !main_fnsig.bound_vars().is_empty() {
        let generics_param_span = main_fn_generics_params_span(tcx, main_def_id);
        tcx.dcx().emit_err(errors::MainFunctionGenericParameters {
            span: generics_param_span.unwrap_or(main_span),
            label_span: generics_param_span,
        });
    } else if !main_fn_predicates.predicates.is_empty() {
        // generics may bring in implicit predicates, so we skip this check if generics is present.
        let generics_where_clauses_span = main_fn_where_clauses_span(tcx, main_def_id);
        tcx.dcx().emit_err(errors::WhereClauseOnMain {
            span: generics_where_clauses_span.unwrap_or(main_span),
            generics_span: generics_where_clauses_span,
        });
    }
}
