use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg};
use clippy_utils::source::snippet;
use clippy_utils::ty::implements_trait;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::FILTER_NEXT;

/// lint use of `filter().next()` for `Iterators`
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    recv: &'tcx hir::Expr<'_>,
    filter_arg: &'tcx hir::Expr<'_>,
) {
    // lint if caller of `.filter().next()` is an Iterator
    let recv_impls_iterator = cx.tcx.get_diagnostic_item(sym::Iterator).map_or(false, |id| {
        implements_trait(cx, cx.typeck_results().expr_ty(recv), id, &[])
    });
    if recv_impls_iterator {
        let msg = "called `filter(..).next()` on an `Iterator`. This is more succinctly expressed by calling \
                   `.find(..)` instead";
        let filter_snippet = snippet(cx, filter_arg.span, "..");
        if filter_snippet.lines().count() <= 1 {
            let iter_snippet = snippet(cx, recv.span, "..");
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
