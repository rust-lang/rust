use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_trait_method;
use clippy_utils::ty::implements_trait;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_span::{Span, sym};

use super::DOUBLE_ENDED_ITERATOR_LAST;

pub(super) fn check(cx: &LateContext<'_>, expr: &'_ Expr<'_>, self_expr: &'_ Expr<'_>, call_span: Span) {
    if is_trait_method(cx, expr, sym::Iterator)
        && let Some(deiter_id) = cx.tcx.get_diagnostic_item(sym::DoubleEndedIterator)
        && implements_trait(cx, cx.typeck_results().expr_ty(self_expr).peel_refs(), deiter_id, &[])
    {
        span_lint_and_sugg(
            cx,
            DOUBLE_ENDED_ITERATOR_LAST,
            call_span,
            "called `Iterator::last` on a `DoubleEndedIterator`; this will needlessly iterate the entire iterator",
            "try",
            "next_back()".to_string(),
            Applicability::MachineApplicable,
        );
    }
}
