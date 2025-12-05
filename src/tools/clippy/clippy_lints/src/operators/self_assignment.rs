use clippy_utils::diagnostics::span_lint;
use clippy_utils::eq_expr_value;
use clippy_utils::source::snippet;
use rustc_hir::Expr;
use rustc_lint::LateContext;

use super::SELF_ASSIGNMENT;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>, lhs: &'tcx Expr<'_>, rhs: &'tcx Expr<'_>) {
    if eq_expr_value(cx, lhs, rhs) {
        let lhs = snippet(cx, lhs.span, "<lhs>");
        let rhs = snippet(cx, rhs.span, "<rhs>");
        span_lint(
            cx,
            SELF_ASSIGNMENT,
            e.span,
            format!("self-assignment of `{rhs}` to `{lhs}`"),
        );
    }
}
