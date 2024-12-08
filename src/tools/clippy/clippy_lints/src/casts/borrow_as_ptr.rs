use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_context;
use clippy_utils::std_or_core;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, Mutability, Ty, TyKind};
use rustc_lint::LateContext;
use rustc_middle::ty::adjustment::Adjust;

use super::BORROW_AS_PTR;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    cast_expr: &'tcx Expr<'_>,
    cast_to: &'tcx Ty<'_>,
) {
    if matches!(cast_to.kind, TyKind::Ptr(_))
        && let ExprKind::AddrOf(BorrowKind::Ref, mutability, e) = cast_expr.kind
        && let Some(std_or_core) = std_or_core(cx)
    {
        let macro_name = match mutability {
            Mutability::Not => "addr_of",
            Mutability::Mut => "addr_of_mut",
        };
        let mut app = Applicability::MachineApplicable;
        let snip = snippet_with_context(cx, e.span, cast_expr.span.ctxt(), "..", &mut app).0;
        // Fix #9884
        if !e.is_place_expr(|base| {
            cx.typeck_results()
                .adjustments()
                .get(base.hir_id)
                .is_some_and(|x| x.iter().any(|adj| matches!(adj.kind, Adjust::Deref(_))))
        }) {
            return;
        }

        span_lint_and_sugg(
            cx,
            BORROW_AS_PTR,
            expr.span,
            "borrow as raw pointer",
            "try",
            format!("{std_or_core}::ptr::{macro_name}!({snip})"),
            Applicability::MachineApplicable,
        );
    }
}
