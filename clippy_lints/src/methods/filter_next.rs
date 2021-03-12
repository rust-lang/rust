use crate::utils::{match_trait_method, paths, snippet, span_lint, span_lint_and_sugg};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;

use super::FILTER_NEXT;

/// lint use of `filter().next()` for `Iterators`
pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>, filter_args: &'tcx [hir::Expr<'_>]) {
    // lint if caller of `.filter().next()` is an Iterator
    if match_trait_method(cx, expr, &paths::ITERATOR) {
        let msg = "called `filter(..).next()` on an `Iterator`. This is more succinctly expressed by calling \
                   `.find(..)` instead";
        let filter_snippet = snippet(cx, filter_args[1].span, "..");
        if filter_snippet.lines().count() <= 1 {
            let iter_snippet = snippet(cx, filter_args[0].span, "..");
            // add note if not multi-line
            span_lint_and_sugg(
                cx,
                FILTER_NEXT,
                expr.span,
                msg,
                "try this",
                format!("{}.find({})", iter_snippet, filter_snippet),
                Applicability::MachineApplicable,
            );
        } else {
            span_lint(cx, FILTER_NEXT, expr.span, msg);
        }
    }
}
