use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{is_res_lang_ctor, path_res};
use rustc_errors::Applicability;
use rustc_hir::LangItem::{ResultErr, ResultOk};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use super::OPTION_MAP_OR_ERR_OK;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    recv: &'tcx Expr<'_>,
    or_expr: &'tcx Expr<'_>,
    map_expr: &'tcx Expr<'_>,
) {
    // We check that it's called on an `Option` type.
    if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(recv), sym::Option)
        // We check that first we pass an `Err`.
        && let ExprKind::Call(call, &[arg]) = or_expr.kind
        && is_res_lang_ctor(cx, path_res(cx, call), ResultErr)
        // And finally we check that it is mapped as `Ok`.
        && is_res_lang_ctor(cx, path_res(cx, map_expr), ResultOk)
    {
        let msg = "called `map_or(Err(_), Ok)` on an `Option` value";
        let self_snippet = snippet(cx, recv.span, "..");
        let err_snippet = snippet(cx, arg.span, "..");
        span_lint_and_sugg(
            cx,
            OPTION_MAP_OR_ERR_OK,
            expr.span,
            msg,
            "try using `ok_or` instead",
            format!("{self_snippet}.ok_or({err_snippet})"),
            Applicability::MachineApplicable,
        );
    }
}
