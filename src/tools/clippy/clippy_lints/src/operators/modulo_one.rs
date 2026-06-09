use clippy_utils::diagnostics::span_lint;
use clippy_utils::{is_integer_const, unsext};
use rustc_hir::{BinOpKind, Expr};
use rustc_lint::LateContext;
use rustc_middle::ty;

use super::MODULO_ONE;

pub(crate) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, op: BinOpKind, right: &Expr<'_>) {
    if op == BinOpKind::Rem {
        if is_integer_const(cx, right, 1) {
            span_lint(cx, MODULO_ONE, expr.span, "any number modulo 1 will be 0");
        }

        if let ty::Int(ity) = cx.typeck_results().expr_ty(right).kind()
            && is_integer_const(cx, right, unsext(cx.tcx, -1, *ity))
        {
            span_lint(
                cx,
                MODULO_ONE,
                expr.span,
                "any number modulo -1 will panic/overflow or result in 0",
            );
        }
    }
}
