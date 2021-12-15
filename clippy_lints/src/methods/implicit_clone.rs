use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::{is_diag_item_method, is_diag_trait_item};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty::TyS;
use rustc_span::{sym, Span};

use super::IMPLICIT_CLONE;

pub fn check(cx: &LateContext<'_>, method_name: &str, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>, span: Span) {
    if_chain! {
        if let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
        if is_clone_like(cx, method_name, method_def_id);
        let return_type = cx.typeck_results().expr_ty(expr);
        let input_type = cx.typeck_results().expr_ty(recv).peel_refs();
        if let Some(ty_name) = input_type.ty_adt_def().map(|adt_def| cx.tcx.item_name(adt_def.did));
        if TyS::same_type(return_type, input_type);
        then {
            span_lint_and_sugg(
                cx,
                IMPLICIT_CLONE,
                span,
                &format!("implicitly cloning a `{}` by calling `{}` on its dereferenced type", ty_name, method_name),
                "consider using",
                "clone".to_string(),
                Applicability::MachineApplicable
            );
        }
    }
}

/// Returns true if the named method can be used to clone the receiver.
/// Note that `to_string` is not flagged by `implicit_clone`. So other lints that call
/// `is_clone_like` and that do flag `to_string` must handle it separately. See, e.g.,
/// `is_to_owned_like` in `unnecessary_to_owned.rs`.
pub fn is_clone_like(cx: &LateContext<'_>, method_name: &str, method_def_id: hir::def_id::DefId) -> bool {
    match method_name {
        "to_os_string" => is_diag_item_method(cx, method_def_id, sym::OsStr),
        "to_owned" => is_diag_trait_item(cx, method_def_id, sym::ToOwned),
        "to_path_buf" => is_diag_item_method(cx, method_def_id, sym::Path),
        "to_vec" => {
            cx.tcx
                .impl_of_method(method_def_id)
                .map(|impl_did| Some(impl_did) == cx.tcx.lang_items().slice_alloc_impl())
                == Some(true)
        },
        _ => false,
    }
}
