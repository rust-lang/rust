use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::SpanRangeExt;
use clippy_utils::std_or_core;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

use super::PTR_EQ;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    op: BinOpKind,
    left: &'tcx Expr<'_>,
    right: &'tcx Expr<'_>,
) {
    if BinOpKind::Eq == op {
        // Remove one level of usize conversion if any
        let (left, right) = match (expr_as_cast_to_usize(cx, left), expr_as_cast_to_usize(cx, right)) {
            (Some(lhs), Some(rhs)) => (lhs, rhs),
            _ => (left, right),
        };

        // This lint concerns raw pointers
        let (left_ty, right_ty) = (cx.typeck_results().expr_ty(left), cx.typeck_results().expr_ty(right));
        if !left_ty.is_raw_ptr() || !right_ty.is_raw_ptr() {
            return;
        }

        let (left_var, right_var) = (peel_raw_casts(cx, left, left_ty), peel_raw_casts(cx, right, right_ty));

        if let Some(left_snip) = left_var.span.get_source_text(cx)
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

// Peel raw casts if the remaining expression can be coerced to it
fn peel_raw_casts<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, expr_ty: Ty<'tcx>) -> &'tcx Expr<'tcx> {
    if let ExprKind::Cast(inner, _) = expr.kind
        && let ty::RawPtr(target_ty, _) = expr_ty.kind()
        && let inner_ty = cx.typeck_results().expr_ty(inner)
        && let ty::RawPtr(inner_target_ty, _) | ty::Ref(_, inner_target_ty, _) = inner_ty.kind()
        && target_ty == inner_target_ty
    {
        peel_raw_casts(cx, inner, inner_ty)
    } else {
        expr
    }
}
