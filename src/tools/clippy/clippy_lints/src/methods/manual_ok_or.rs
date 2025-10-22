use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::res::{MaybeDef, MaybeQPath, MaybeResPath};
use clippy_utils::source::{SpanRangeExt, indent_of, reindent_multiline};
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
    if let Some(method_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && let Some(impl_id) = cx.tcx.impl_of_assoc(method_id)
        && cx
            .tcx
            .type_of(impl_id)
            .instantiate_identity()
            .is_diag_item(cx, sym::Option)
        && let ExprKind::Call(err_path, [err_arg]) = or_expr.kind
        && err_path.res(cx).ctor_parent(cx).is_lang_item(cx, ResultErr)
        && is_ok_wrapping(cx, map_expr)
        && let Some(recv_snippet) = recv.span.get_source_text(cx)
        && let Some(err_arg_snippet) = err_arg.span.get_source_text(cx)
        && let Some(indent) = indent_of(cx, expr.span)
    {
        let reindented_err_arg_snippet = reindent_multiline(err_arg_snippet.as_str(), true, Some(indent + 4));
        span_lint_and_sugg(
            cx,
            MANUAL_OK_OR,
            expr.span,
            "this pattern reimplements `Option::ok_or`",
            "replace with",
            format!("{recv_snippet}.ok_or({reindented_err_arg_snippet})"),
            Applicability::MachineApplicable,
        );
    }
}

fn is_ok_wrapping(cx: &LateContext<'_>, map_expr: &Expr<'_>) -> bool {
    match map_expr.kind {
        ExprKind::Path(ref qpath)
            if cx
                .qpath_res(qpath, map_expr.hir_id)
                .ctor_parent(cx)
                .is_lang_item(cx, ResultOk) =>
        {
            true
        },
        ExprKind::Closure(closure) => {
            let body = cx.tcx.hir_body(closure.body);
            if let PatKind::Binding(_, param_id, ..) = body.params[0].pat.kind
                && let ExprKind::Call(callee, [ok_arg]) = body.value.kind
                && callee.res(cx).ctor_parent(cx).is_lang_item(cx, ResultOk)
            {
                ok_arg.res_local_id() == Some(param_id)
            } else {
                false
            }
        },
        _ => false,
    }
}
