use super::ITER_NEXT_LOOP;
use crate::utils::{match_trait_method, paths, span_lint};
use rustc_hir::Expr;
use rustc_lint::LateContext;

pub(super) fn check(cx: &LateContext<'_>, arg: &Expr<'_>, expr: &Expr<'_>) -> bool {
    if match_trait_method(cx, arg, &paths::ITERATOR) {
        span_lint(
            cx,
            ITER_NEXT_LOOP,
            expr.span,
            "you are iterating over `Iterator::next()` which is an Option; this will compile but is \
            probably not what you want",
        );
        true
    } else {
        false
    }
}
