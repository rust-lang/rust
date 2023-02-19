use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::Node;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::config::EntryFnType;
use rustc_span::def_id::{DefId, LocalDefId, CRATE_DEF_ID};
use rustc_span::{symbol::sym, Span};
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::traits::error_reporting::TypeErrCtxtExt as _;
use rustc_trait_selection::traits::{self, ObligationCause, ObligationCauseCode};

use std::ops::Not;

use crate::require_same_types;

pub(crate) fn check_for_entry_fn(tcx: TyCtxt<'_>) {
    match tcx.entry_fn(()) {
        Some((def_id, EntryFnType::Main { .. })) => check_main_fn_ty(tcx, def_id),
        Some((def_id, EntryFnType::Start)) => check_start_fn_ty(tcx, def_id),
        _ => {}
    }
}

fn check_main_fn_ty(tcx: TyCtxt<'_>, main_def_id: DefId) {
    let main_fnsig = tcx.fn_sig(main_def_id).subst_identity();
    let main_span = tcx.def_span(main_def_id);

    fn main_fn_diagnostics_def_id(tcx: TyCtxt<'_>, def_id: DefId, sp: Span) -> LocalDefId {
        if let Some(local_def_id) = def_id.as_local() {
            let hir_type = tcx.type_of(local_def_id).subst_identity();
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
        let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
        match tcx.hir().find(hir_id) {
            Some(Node::Item(hir::Item { kind: hir::ItemKind::Fn(_, generics, _), .. })) => {
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
        let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
        match tcx.hir().find(hir_id) {
            Some(Node::Item(hir::Item { kind: hir::ItemKind::Fn(_, generics, _), .. })) => {
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
        let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
        match tcx.hir().find(hir_id) {
            Some(Node::Item(hir::Item { kind: hir::ItemKind::Fn(fn_sig, _, _), .. })) => {
                Some(fn_sig.decl.output.span())
            }
            _ => {
                span_bug!(tcx.def_span(def_id), "main has a non-function type");
            }
        }
    }

    let mut error = false;
    let main_diagnostics_def_id = main_fn_diagnostics_def_id(tcx, main_def_id, main_span);
    let main_fn_generics = tcx.generics_of(main_def_id);
    let main_fn_predicates = tcx.predicates_of(main_def_id);
    if main_fn_generics.count() != 0 || !main_fnsig.bound_vars().is_empty() {
        let generics_param_span = main_fn_generics_params_span(tcx, main_def_id);
        let msg = "`main` function is not allowed to have generic \
            parameters";
        let mut diag =
            struct_span_err!(tcx.sess, generics_param_span.unwrap_or(main_span), E0131, "{}", msg);
        if let Some(generics_param_span) = generics_param_span {
            let label = "`main` cannot have generic parameters";
            diag.span_label(generics_param_span, label);
        }
        diag.emit();
        error = true;
    } else if !main_fn_predicates.predicates.is_empty() {
        // generics may bring in implicit predicates, so we skip this check if generics is present.
        let generics_where_clauses_span = main_fn_where_clauses_span(tcx, main_def_id);
        let mut diag = struct_span_err!(
            tcx.sess,
            generics_where_clauses_span.unwrap_or(main_span),
            E0646,
            "`main` function is not allowed to have a `where` clause"
        );
        if let Some(generics_where_clauses_span) = generics_where_clauses_span {
            diag.span_label(generics_where_clauses_span, "`main` cannot have a `where` clause");
        }
        diag.emit();
        error = true;
    }

    let main_asyncness = tcx.asyncness(main_def_id);
    if let hir::IsAsync::Async = main_asyncness {
        let mut diag = struct_span_err!(
            tcx.sess,
            main_span,
            E0752,
            "`main` function is not allowed to be `async`"
        );
        let asyncness_span = main_fn_asyncness_span(tcx, main_def_id);
        if let Some(asyncness_span) = asyncness_span {
            diag.span_label(asyncness_span, "`main` function is not allowed to be `async`");
        }
        diag.emit();
        error = true;
    }

    for attr in tcx.get_attrs(main_def_id, sym::track_caller) {
        tcx.sess
            .struct_span_err(attr.span, "`main` function is not allowed to be `#[track_caller]`")
            .span_label(main_span, "`main` function is not allowed to be `#[track_caller]`")
            .emit();
        error = true;
    }

    if error {
        return;
    }

    let expected_return_type;
    if let Some(term_did) = tcx.lang_items().termination() {
        let return_ty = main_fnsig.output();
        let return_ty_span = main_fn_return_type_span(tcx, main_def_id).unwrap_or(main_span);
        if !return_ty.bound_vars().is_empty() {
            let msg = "`main` function return type is not allowed to have generic \
                    parameters";
            struct_span_err!(tcx.sess, return_ty_span, E0131, "{}", msg).emit();
            error = true;
        }
        let return_ty = return_ty.skip_binder();
        let infcx = tcx.infer_ctxt().build();
        // Main should have no WC, so empty param env is OK here.
        let param_env = ty::ParamEnv::empty();
        let cause = traits::ObligationCause::new(
            return_ty_span,
            main_diagnostics_def_id,
            ObligationCauseCode::MainFunctionType,
        );
        let ocx = traits::ObligationCtxt::new(&infcx);
        let norm_return_ty = ocx.normalize(&cause, param_env, return_ty);
        ocx.register_bound(cause, param_env, norm_return_ty, term_did);
        let errors = ocx.select_all_or_error();
        if !errors.is_empty() {
            infcx.err_ctxt().report_fulfillment_errors(&errors, None);
            error = true;
        }
        // now we can take the return type of the given main function
        expected_return_type = main_fnsig.output();
    } else {
        // standard () main return type
        expected_return_type = ty::Binder::dummy(tcx.mk_unit());
    }

    if error {
        return;
    }

    let se_ty = tcx.mk_fn_ptr(expected_return_type.map_bound(|expected_return_type| {
        tcx.mk_fn_sig([], expected_return_type, false, hir::Unsafety::Normal, Abi::Rust)
    }));

    require_same_types(
        tcx,
        &ObligationCause::new(
            main_span,
            main_diagnostics_def_id,
            ObligationCauseCode::MainFunctionType,
        ),
        se_ty,
        tcx.mk_fn_ptr(main_fnsig),
    );
}

fn check_start_fn_ty(tcx: TyCtxt<'_>, start_def_id: DefId) {
    let start_def_id = start_def_id.expect_local();
    let start_id = tcx.hir().local_def_id_to_hir_id(start_def_id);
    let start_span = tcx.def_span(start_def_id);
    let start_t = tcx.type_of(start_def_id).subst_identity();
    match start_t.kind() {
        ty::FnDef(..) => {
            if let Some(Node::Item(it)) = tcx.hir().find(start_id) {
                if let hir::ItemKind::Fn(sig, generics, _) = &it.kind {
                    let mut error = false;
                    if !generics.params.is_empty() {
                        struct_span_err!(
                            tcx.sess,
                            generics.span,
                            E0132,
                            "start function is not allowed to have type parameters"
                        )
                        .span_label(generics.span, "start function cannot have type parameters")
                        .emit();
                        error = true;
                    }
                    if generics.has_where_clause_predicates {
                        struct_span_err!(
                            tcx.sess,
                            generics.where_clause_span,
                            E0647,
                            "start function is not allowed to have a `where` clause"
                        )
                        .span_label(
                            generics.where_clause_span,
                            "start function cannot have a `where` clause",
                        )
                        .emit();
                        error = true;
                    }
                    if let hir::IsAsync::Async = sig.header.asyncness {
                        let span = tcx.def_span(it.owner_id);
                        struct_span_err!(
                            tcx.sess,
                            span,
                            E0752,
                            "`start` is not allowed to be `async`"
                        )
                        .span_label(span, "`start` is not allowed to be `async`")
                        .emit();
                        error = true;
                    }

                    let attrs = tcx.hir().attrs(start_id);
                    for attr in attrs {
                        if attr.has_name(sym::track_caller) {
                            tcx.sess
                                .struct_span_err(
                                    attr.span,
                                    "`start` is not allowed to be `#[track_caller]`",
                                )
                                .span_label(
                                    start_span,
                                    "`start` is not allowed to be `#[track_caller]`",
                                )
                                .emit();
                            error = true;
                        }
                    }

                    if error {
                        return;
                    }
                }
            }

            let se_ty = tcx.mk_fn_ptr(ty::Binder::dummy(tcx.mk_fn_sig(
                [tcx.types.isize, tcx.mk_imm_ptr(tcx.mk_imm_ptr(tcx.types.u8))],
                tcx.types.isize,
                false,
                hir::Unsafety::Normal,
                Abi::Rust,
            )));

            require_same_types(
                tcx,
                &ObligationCause::new(
                    start_span,
                    start_def_id,
                    ObligationCauseCode::StartFunctionType,
                ),
                se_ty,
                tcx.mk_fn_ptr(tcx.fn_sig(start_def_id).subst_identity()),
            );
        }
        _ => {
            span_bug!(start_span, "start has a non-function type: found `{}`", start_t);
        }
    }
}
