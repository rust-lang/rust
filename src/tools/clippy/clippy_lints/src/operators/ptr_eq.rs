use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::SpanRangeExt;
use clippy_utils::std_or_core;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;

use super::PTR_EQ;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    op: BinOpKind,
    left: &'tcx Expr<'_>,
    right: &'tcx Expr<'_>,
) {
    if BinOpKind::Eq == op {
        let (left, right) = match (expr_as_cast_to_usize(cx, left), expr_as_cast_to_usize(cx, right)) {
            (Some(lhs), Some(rhs)) => (lhs, rhs),
            _ => (left, right),
        };

        if let Some(left_var) = expr_as_cast_to_raw_pointer(cx, left)
            && let Some(right_var) = expr_as_cast_to_raw_pointer(cx, right)
            && let Some(left_snip) = left_var.span.get_source_text(cx)
            && let Some(right_snip) = right_var.span.get_source_text(cx)
        {
            let Some(top_crate) = std_or_core(cx) else { return };
            span_lint_and_sugg(
                cx,
                PTR_EQ,
                expr.span,
                format!("use `{top_crate}::ptr::eq` when comparing raw pointers"),
                "try",
                format!("{top_crate}::ptr::eq({left_snip}, {right_snip})"),
                Applicability::MachineApplicable,
            );
        }
    }
}

// If the given expression is a cast to a usize, return the lhs of the cast
// E.g., `foo as *const _ as usize` returns `foo as *const _`.
fn expr_as_cast_to_usize<'tcx>(cx: &LateContext<'tcx>, cast_expr: &'tcx Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
    if cx.typeck_results().expr_ty(cast_expr) == cx.tcx.types.usize {
        if let ExprKind::Cast(expr, _) = cast_expr.kind {
            return Some(expr);
        }
    }
    None
}

// If the given expression is a cast to a `*const` pointer, return the lhs of the cast
// E.g., `foo as *const _` returns `foo`.
fn expr_as_cast_to_raw_pointer<'tcx>(cx: &LateContext<'tcx>, cast_expr: &'tcx Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
    if cx.typeck_results().expr_ty(cast_expr).is_raw_ptr() {
        if let ExprKind::Cast(expr, _) = cast_expr.kind {
            return Some(expr);
        }
    }
    None
}
