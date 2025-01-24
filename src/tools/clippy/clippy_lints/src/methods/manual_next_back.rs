use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_trait_method;
use clippy_utils::ty::implements_trait;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    rev_call: &'tcx Expr<'_>,
    rev_recv: &'tcx Expr<'_>,
) {
    let rev_recv_ty = cx.typeck_results().expr_ty(rev_recv);

    // check that the receiver of `rev` implements `DoubleEndedIterator` and
    // that `rev` and `next` come from `Iterator`
    if cx
        .tcx
        .get_diagnostic_item(sym::DoubleEndedIterator)
        .is_some_and(|double_ended_iterator| implements_trait(cx, rev_recv_ty, double_ended_iterator, &[]))
        && is_trait_method(cx, rev_call, sym::Iterator)
        && is_trait_method(cx, expr, sym::Iterator)
    {
        span_lint_and_sugg(
            cx,
            super::MANUAL_NEXT_BACK,
            expr.span.with_lo(rev_recv.span.hi()),
            "manual backwards iteration",
            "use",
            String::from(".next_back()"),
            Applicability::MachineApplicable,
        );
    }
}
