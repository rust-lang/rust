use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{indent_of, reindent_multiline, snippet_opt};
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{is_res_lang_ctor, path_res, path_to_local_id};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::LangItem::{ResultErr, ResultOk};
use rustc_hir::{Expr, ExprKind, PatKind};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use super::MANUAL_OK_OR;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    recv: &'tcx Expr<'_>,
    or_expr: &'tcx Expr<'_>,
    map_expr: &'tcx Expr<'_>,
) {
    if_chain! {
        if let Some(method_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
        if let Some(impl_id) = cx.tcx.impl_of_method(method_id);
        if is_type_diagnostic_item(cx, cx.tcx.bound_type_of(impl_id).subst_identity(), sym::Option);
        if let ExprKind::Call(err_path, [err_arg]) = or_expr.kind;
        if is_res_lang_ctor(cx, path_res(cx, err_path), ResultErr);
        if is_ok_wrapping(cx, map_expr);
        if let Some(recv_snippet) = snippet_opt(cx, recv.span);
        if let Some(err_arg_snippet) = snippet_opt(cx, err_arg.span);
        if let Some(indent) = indent_of(cx, expr.span);
        then {
            let reindented_err_arg_snippet = reindent_multiline(err_arg_snippet.into(), true, Some(indent + 4));
            span_lint_and_sugg(
                cx,
                MANUAL_OK_OR,
                expr.span,
                "this pattern reimplements `Option::ok_or`",
                "replace with",
                format!(
                    "{recv_snippet}.ok_or({reindented_err_arg_snippet})"
                ),
                Applicability::MachineApplicable,
            );
        }
    }
}

fn is_ok_wrapping(cx: &LateContext<'_>, map_expr: &Expr<'_>) -> bool {
    match map_expr.kind {
        ExprKind::Path(ref qpath) if is_res_lang_ctor(cx, cx.qpath_res(qpath, map_expr.hir_id), ResultOk) => true,
        ExprKind::Closure(closure) => {
            let body = cx.tcx.hir().body(closure.body);
            if let PatKind::Binding(_, param_id, ..) = body.params[0].pat.kind
                && let ExprKind::Call(callee, [ok_arg]) = body.value.kind
                && is_res_lang_ctor(cx, path_res(cx, callee), ResultOk)
            {
                path_to_local_id(ok_arg, param_id)
            } else {
                false
            }
        },
        _ => false,
    }
}
