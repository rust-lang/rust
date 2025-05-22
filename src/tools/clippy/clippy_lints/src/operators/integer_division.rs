use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use super::INTEGER_DIVISION;

pub(crate) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    op: hir::BinOpKind,
    left: &'tcx hir::Expr<'_>,
    right: &'tcx hir::Expr<'_>,
) {
    if op == hir::BinOpKind::Div
        && cx.typeck_results().expr_ty(left).is_integral()
        && let right_ty = cx.typeck_results().expr_ty(right)
        && (right_ty.is_integral() || is_type_diagnostic_item(cx, right_ty, sym::NonZero))
    {
        #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
        span_lint_and_then(cx, INTEGER_DIVISION, expr.span, "integer division", |diag| {
            diag.help("division of integers may cause loss of precision. consider using floats");
        });
    }
}
