use super::ITER_NEXT_LOOP;
use crate::utils::span_lint;
use rustc_hir::Expr;
use rustc_lint::LateContext;

pub(super) fn lint(cx: &LateContext<'_>, expr: &Expr<'_>) {
    span_lint(
        cx,
        ITER_NEXT_LOOP,
        expr.span,
        "you are iterating over `Iterator::next()` which is an Option; this will compile but is \
        probably not what you want",
    );
}
