use rustc_hir as hir;
use rustc_hir::Node;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::config::EntryFnType;
use rustc_span::def_id::{DefId, LocalDefId, CRATE_DEF_ID};
use rustc_span::{symbol::sym, Span};
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::traits::error_reporting::TypeErrCtxtExt as _;
use rustc_trait_selection::traits::{self, ObligationCause, ObligationCauseCode};

use std::ops::Not;

use crate::errors;
use crate::require_same_types;

pub(crate) fn check_for_entry_fn(tcx: TyCtxt<'_>) {
    match tcx.entry_fn(()) {
        Some((def_id, EntryFnType::Main { .. })) => check_main_fn_ty(tcx, def_id),
        Some((def_id, EntryFnType::Start)) => check_start_fn_ty(tcx, def_id),
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
        tcx.sess.emit_err(errors::MainFunctionGenericParameters {
            span: generics_param_span.unwrap_or(main_span),
            label_span: generics_param_span,
        });
        error = true;
    } else if !main_fn_predicates.predicates.is_empty() {
        // generics may bring in implicit predicates, so we skip this check if generics is present.
        let generics_where_clauses_span = main_fn_where_clauses_span(tcx, main_def_id);
        tcx.sess.emit_err(errors::WhereClauseOnMain {
            span: generics_where_clauses_span.unwrap_or(main_span),
            generics_span: generics_where_clauses_span,
        });
        error = true;
    }

    let main_asyncness = tcx.asyncness(main_def_id);
    if let hir::IsAsync::Async = main_asyncness {
        let asyncness_span = main_fn_asyncness_span(tcx, main_def_id);
        tcx.sess.emit_err(errors::MainFunctionAsync { span: main_span, asyncness: asyncness_span });
        error = true;
    }

    for attr in tcx.get_attrs(main_def_id, sym::track_caller) {
        tcx.sess.emit_err(errors::TrackCallerOnMain { span: attr.span, annotated: main_span });
        error = true;
    }

    if !tcx.codegen_fn_attrs(main_def_id).target_features.is_empty()
        // Calling functions with `#[target_feature]` is not unsafe on WASM, see #84988
        && !tcx.sess.target.is_like_wasm
        && !tcx.sess.opts.actually_rustdoc
    {
        tcx.sess.emit_err(errors::TargetFeatureOnMain { main: main_span });
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
        if !return_ty.bound_vars().is_empty() {
            tcx.sess.emit_err(errors::MainFunctionReturnTypeGeneric { span: return_ty_span });
            error = true;
        }
        let return_ty = return_ty.skip_binder();
        let infcx = tcx.infer_ctxt().build();
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
            infcx.err_ctxt().report_fulfillment_errors(&errors);
            error = true;
        }
        // now we can take the return type of the given main function
        expected_return_type = main_fnsig.output();
    } else {
        // standard () main return type
        expected_return_type = ty::Binder::dummy(Ty::new_unit(tcx));
    }

    if error {
        return;
    }

    let se_ty = Ty::new_fn_ptr(
        tcx,
        expected_return_type.map_bound(|expected_return_type| {
            tcx.mk_fn_sig([], expected_return_type, false, hir::Unsafety::Normal, Abi::Rust)
        }),
    );

    require_same_types(
        tcx,
        &ObligationCause::new(
            main_span,
            main_diagnostics_def_id,
            ObligationCauseCode::MainFunctionType,
        ),
        param_env,
        se_ty,
        Ty::new_fn_ptr(tcx, main_fnsig),
    );
}

fn check_start_fn_ty(tcx: TyCtxt<'_>, start_def_id: DefId) {
    let start_def_id = start_def_id.expect_local();
    let start_id = tcx.hir().local_def_id_to_hir_id(start_def_id);
    let start_span = tcx.def_span(start_def_id);
    let start_t = tcx.type_of(start_def_id).instantiate_identity();
    match start_t.kind() {
        ty::FnDef(..) => {
            if let Some(Node::Item(it)) = tcx.hir().find(start_id) {
                if let hir::ItemKind::Fn(sig, generics, _) = &it.kind {
                    let mut error = false;
                    if !generics.params.is_empty() {
                        tcx.sess.emit_err(errors::StartFunctionParameters { span: generics.span });
                        error = true;
                    }
                    if generics.has_where_clause_predicates {
                        tcx.sess.emit_err(errors::StartFunctionWhere {
                            span: generics.where_clause_span,
                        });
                        error = true;
                    }
                    if let hir::IsAsync::Async = sig.header.asyncness {
                        let span = tcx.def_span(it.owner_id);
                        tcx.sess.emit_err(errors::StartAsync { span: span });
                        error = true;
                    }

                    let attrs = tcx.hir().attrs(start_id);
                    for attr in attrs {
                        if attr.has_name(sym::track_caller) {
                            tcx.sess.emit_err(errors::StartTrackCaller {
                                span: attr.span,
                                start: start_span,
                            });
                            error = true;
                        }
                        if attr.has_name(sym::target_feature)
                            // Calling functions with `#[target_feature]` is
                            // not unsafe on WASM, see #84988
                            && !tcx.sess.target.is_like_wasm
                            && !tcx.sess.opts.actually_rustdoc
                        {
                            tcx.sess.emit_err(errors::StartTargetFeature {
                                span: attr.span,
                                start: start_span,
                            });
                            error = true;
                        }
                    }

                    if error {
                        return;
                    }
                }
            }

            let se_ty = Ty::new_fn_ptr(
                tcx,
                ty::Binder::dummy(tcx.mk_fn_sig(
                    [tcx.types.isize, Ty::new_imm_ptr(tcx, Ty::new_imm_ptr(tcx, tcx.types.u8))],
                    tcx.types.isize,
                    false,
                    hir::Unsafety::Normal,
                    Abi::Rust,
                )),
            );

            require_same_types(
                tcx,
                &ObligationCause::new(
                    start_span,
                    start_def_id,
                    ObligationCauseCode::StartFunctionType,
                ),
                ty::ParamEnv::empty(), // start should not have any where bounds.
                se_ty,
                Ty::new_fn_ptr(tcx, tcx.fn_sig(start_def_id).instantiate_identity()),
            );
        }
        _ => {
            span_bug!(start_span, "start has a non-function type: found `{}`", start_t);
        }
    }
}
