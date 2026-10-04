use clippy_utils::consts::{FullInt, eval_int};
use clippy_utils::diagnostics::span_lint;
use rustc_hir::{BinOpKind, Expr};
use rustc_lint::LateContext;

use super::MODULO_ONE;

pub(crate) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, op: BinOpKind, right: &Expr<'_>) {
    if op == BinOpKind::Rem {
        let msg = match eval_int(cx, right) {
            Some(FullInt::S(-1)) => "any number modulo -1 will panic/overflow or result in 0",
            Some(FullInt::U(1)) => "any number modulo 1 will be 0",
            _ => return,
        };
        span_lint(cx, MODULO_ONE, expr.span, msg);
    }
}
