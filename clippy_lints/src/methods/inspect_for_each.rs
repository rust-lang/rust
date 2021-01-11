use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::source_map::Span;

use crate::utils::{match_trait_method, paths, span_lint_and_help};

use super::INSPECT_FOR_EACH;

/// lint use of `inspect().for_each()` for `Iterators`
pub(super) fn lint<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>, inspect_span: Span) {
    if match_trait_method(cx, expr, &paths::ITERATOR) {
        let msg = "called `inspect(..).for_each(..)` on an `Iterator`";
        let hint = "move the code from `inspect(..)` to `for_each(..)` and remove the `inspect(..)`";
        span_lint_and_help(
            cx,
            INSPECT_FOR_EACH,
            inspect_span.with_hi(expr.span.hi()),
            msg,
            None,
            hint,
        );
    }
}
