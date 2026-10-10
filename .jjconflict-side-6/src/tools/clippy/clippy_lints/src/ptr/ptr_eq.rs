use super::PTR_EQ;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::std_or_core;
use clippy_utils::sugg::Sugg;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    op: BinOpKind,
    left: &'tcx Expr<'_>,
    right: &'tcx Expr<'_>,
    span: Span,
) {
    if span.from_expansion() {
        return;
    }

    // Remove one level of usize conversion if any
    let (left, right, usize_peeled) = match (expr_as_cast_to_usize(cx, left), expr_as_cast_to_usize(cx, right)) {
        (Some(lhs), Some(rhs)) => (lhs, rhs, true),
        _ => (left, right, false),
    };

    // This lint concerns raw pointers
    let (left_ty, right_ty) = (cx.typeck_results().expr_ty(left), cx.typeck_results().expr_ty(right));
    if !left_ty.is_raw_ptr() || !right_ty.is_raw_ptr() {
        return;
    }

    let ((left_var, left_casts_peeled), (right_var, right_casts_peeled)) =
        (peel_raw_casts(cx, left, left_ty), peel_raw_casts(cx, right, right_ty));

    if !(usize_peeled || left_casts_peeled || right_casts_peeled) {
        return;
    }

    let mut app = Applicability::MachineApplicable;
    let ctxt = span.ctxt();
    let left_snip = Sugg::hir_with_context(cx, left_var, ctxt, "_", &mut app);
    let right_snip = Sugg::hir_with_context(cx, right_var, ctxt, "_", &mut app);
    {
        let Some(top_crate) = std_or_core(cx) else { return };
        let invert = if op == BinOpKind::Eq { "" } else { "!" };
        span_lint_and_sugg(
            cx,
            PTR_EQ,
            span,
            format!("use `{top_crate}::ptr::eq` when comparing raw pointers"),
            "try",
            format!("{invert}{top_crate}::ptr::eq({left_snip}, {right_snip})"),
            app,
        );
    }
}

// If the given expression is a cast to a usize, return the lhs of the cast
// E.g., `foo as *const _ as usize` returns `foo as *const _`.
fn expr_as_cast_to_usize<'tcx>(cx: &LateContext<'tcx>, cast_expr: &'tcx Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
    if !cast_expr.span.from_expansion()
        && cx.typeck_results().expr_ty(cast_expr) == cx.tcx.types.usize
        && let ExprKind::Cast(expr, _) = cast_expr.kind
    {
        Some(expr)
    } else {
        None
    }
}

// Peel raw casts if the remaining expression can be coerced to it, and whether casts have been
// peeled or not.
fn peel_raw_casts<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, expr_ty: Ty<'tcx>) -> (&'tcx Expr<'tcx>, bool) {
    if !expr.span.from_expansion()
        && let ExprKind::Cast(inner, _) = expr.kind
        && let ty::RawPtr(target_ty, _) = expr_ty.kind()
        && let inner_ty = cx.typeck_results().expr_ty(inner)
        && let ty::RawPtr(inner_target_ty, _) | ty::Ref(_, inner_target_ty, _) = inner_ty.kind()
        && target_ty == inner_target_ty
    {
        (peel_raw_casts(cx, inner, inner_ty).0, true)
    } else {
        (expr, false)
    }
}
