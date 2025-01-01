use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_trait_method;
use clippy_utils::ty::implements_trait;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_span::{Span, sym};

use super::DOUBLE_ENDED_ITERATOR_LAST;

pub(super) fn check(cx: &LateContext<'_>, expr: &'_ Expr<'_>, self_expr: &'_ Expr<'_>, call_span: Span) {
    let typeck = cx.typeck_results();

    // Check if the current "last" method is that of the Iterator trait
    if !is_trait_method(cx, expr, sym::Iterator) {
        return;
    }

    // Find id for DoubleEndedIterator trait
    let Some(deiter_id) = cx.tcx.get_diagnostic_item(sym::DoubleEndedIterator) else {
        return;
    };

    // Find the type of self
    let self_type = typeck.expr_ty(self_expr).peel_refs();

    // Check that the object implements the DoubleEndedIterator trait
    if !implements_trait(cx, self_type, deiter_id, &[]) {
        return;
    }

    // Emit lint
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
