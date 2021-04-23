use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::is_trait_method;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::FILTER_MAP;

/// lint use of `filter_map().map()` for `Iterators`
pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
    // lint if caller of `.filter_map().map()` is an Iterator
    if is_trait_method(cx, expr, sym::Iterator) {
        let msg = "called `filter_map(..).map(..)` on an `Iterator`";
        let hint = "this is more succinctly expressed by only calling `.filter_map(..)` instead";
        span_lint_and_help(cx, FILTER_MAP, expr.span, msg, None, hint);
    }
}
