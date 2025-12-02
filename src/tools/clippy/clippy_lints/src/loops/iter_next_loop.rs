use super::ITER_NEXT_LOOP;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::res::{MaybeDef, MaybeTypeckRes};
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_span::sym;

pub(super) fn check(cx: &LateContext<'_>, arg: &Expr<'_>) {
    if cx.ty_based_def(arg).opt_parent(cx).is_diag_item(cx, sym::Iterator) {
        span_lint(
            cx,
            ITER_NEXT_LOOP,
            arg.span,
            "you are iterating over `Iterator::next()` which is an Option; this will compile but is \
            probably not what you want",
        );
    }
}
