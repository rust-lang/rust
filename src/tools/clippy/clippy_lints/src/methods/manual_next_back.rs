use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::res::{MaybeDef, MaybeTypeckRes};
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
        && cx.ty_based_def(rev_call).opt_parent(cx).is_diag_item(cx, sym::Iterator)
        && cx.ty_based_def(expr).opt_parent(cx).is_diag_item(cx, sym::Iterator)
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
