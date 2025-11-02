use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::res::MaybeDef;
use clippy_utils::source::snippet_with_context;
use clippy_utils::sym;
use clippy_utils::ty::{implements_trait, peel_and_count_ty_refs};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::Symbol;

use super::IMPLICIT_CLONE;

pub fn check(cx: &LateContext<'_>, method_name: Symbol, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>) {
    if let Some(method_parent_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id).opt_parent(cx)
        && is_clone_like(cx, method_name, method_parent_id)
        && let return_type = cx.typeck_results().expr_ty(expr)
        && let input_type = cx.typeck_results().expr_ty(recv)
        && let (input_type, ref_count, _) = peel_and_count_ty_refs(input_type)
        && !(ref_count > 0 && method_parent_id.is_diag_item(cx, sym::ToOwned))
        && let Some(ty_name) = input_type.ty_adt_def().map(|adt_def| cx.tcx.item_name(adt_def.did()))
        && return_type == input_type
        && let Some(clone_trait) = cx.tcx.lang_items().clone_trait()
        && implements_trait(cx, return_type, clone_trait, &[])
    {
        let mut app = Applicability::MachineApplicable;
        let recv_snip = snippet_with_context(cx, recv.span, expr.span.ctxt(), "..", &mut app).0;
        span_lint_and_sugg(
            cx,
            IMPLICIT_CLONE,
            expr.span,
            format!("implicitly cloning a `{ty_name}` by calling `{method_name}` on its dereferenced type"),
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

/// Returns true if the named method can be used to clone the receiver.
pub fn is_clone_like(cx: &LateContext<'_>, method_name: Symbol, method_parent_id: hir::def_id::DefId) -> bool {
    match method_name {
        sym::to_os_string => method_parent_id.opt_impl_ty(cx).is_diag_item(cx, sym::OsStr),
        sym::to_owned => method_parent_id.is_diag_item(cx, sym::ToOwned),
        sym::to_path_buf => method_parent_id.opt_impl_ty(cx).is_diag_item(cx, sym::Path),
        sym::to_string => method_parent_id.is_diag_item(cx, sym::ToString),
        sym::to_vec => method_parent_id
            .opt_impl_ty(cx)
            .is_some_and(|ty| ty.instantiate_identity().is_slice()),
        _ => false,
    }
}
