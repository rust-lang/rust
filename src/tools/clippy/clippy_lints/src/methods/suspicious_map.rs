use crate::utils::span_lint_and_help;
use rustc_hir as hir;
use rustc_lint::LateContext;

use super::SUSPICIOUS_MAP;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>) {
    span_lint_and_help(
        cx,
        SUSPICIOUS_MAP,
        expr.span,
        "this call to `map()` won't have an effect on the call to `count()`",
        None,
        "make sure you did not confuse `map` with `filter` or `for_each`",
    );
}
