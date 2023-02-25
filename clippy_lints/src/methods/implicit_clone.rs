use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::{implements_trait, peel_mid_ty_refs};
use clippy_utils::{is_diag_item_method, is_diag_trait_item};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::IMPLICIT_CLONE;

pub fn check(cx: &LateContext<'_>, method_name: &str, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>) {
    if_chain! {
        if let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
        if is_clone_like(cx, method_name, method_def_id);
        let return_type = cx.typeck_results().expr_ty(expr);
        let input_type = cx.typeck_results().expr_ty(recv);
        let (input_type, ref_count) = peel_mid_ty_refs(input_type);
        if let Some(ty_name) = input_type.ty_adt_def().map(|adt_def| cx.tcx.item_name(adt_def.did()));
        if return_type == input_type;
        if let Some(clone_trait) = cx.tcx.lang_items().clone_trait();
        if implements_trait(cx, return_type, clone_trait, &[]);
        then {
            let mut app = Applicability::MachineApplicable;
            let recv_snip = snippet_with_context(cx, recv.span, expr.span.ctxt(), "..", &mut app).0;
            span_lint_and_sugg(
                cx,
                IMPLICIT_CLONE,
                expr.span,
                &format!("implicitly cloning a `{ty_name}` by calling `{method_name}` on its dereferenced type"),
                "consider using",
                if ref_count > 1 {
                    format!("({}{recv_snip}).clone()", "*".repeat(ref_count - 1))
                } else {
                    format!("{recv_snip}.clone()")
                },
                app,
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
        "to_vec" => cx
            .tcx
            .impl_of_method(method_def_id)
            .filter(|&impl_did| {
                cx.tcx.type_of(impl_did).subst_identity().is_slice() && cx.tcx.impl_trait_ref(impl_did).is_none()
            })
            .is_some(),
        _ => false,
    }
}
