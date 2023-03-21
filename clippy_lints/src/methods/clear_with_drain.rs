use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher::Range;
use clippy_utils::is_range_full;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_span::symbol::sym;
use rustc_span::Span;

use super::CLEAR_WITH_DRAIN;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, span: Span, arg: &Expr<'_>) {
    let ty = cx.typeck_results().expr_ty(recv);
    if is_type_diagnostic_item(cx, ty, sym::Vec) && let Some(range) = Range::hir(arg) && is_range_full(cx, recv, range)
    {
        span_lint_and_sugg(
            cx,
            CLEAR_WITH_DRAIN,
            span.with_hi(expr.span.hi()),
            "`drain` used to clear a `Vec`",
            "try",
            "clear()".to_string(),
            Applicability::MachineApplicable,
        );
    }
}
