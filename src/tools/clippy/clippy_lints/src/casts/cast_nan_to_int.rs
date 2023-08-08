use super::CAST_NAN_TO_INT;

use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint_and_note;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, cast_expr: &Expr<'_>, from_ty: Ty<'_>, to_ty: Ty<'_>) {
    if from_ty.is_floating_point() && to_ty.is_integral() && is_known_nan(cx, cast_expr) {
        span_lint_and_note(
            cx,
            CAST_NAN_TO_INT,
            expr.span,
            &format!("casting a known NaN to {to_ty}"),
            None,
            "this always evaluates to 0",
        );
    }
}

fn is_known_nan(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    match constant(cx, cx.typeck_results(), e) {
        Some(Constant::F64(n)) => n.is_nan(),
        Some(Constant::F32(n)) => n.is_nan(),
        _ => false,
    }
}
