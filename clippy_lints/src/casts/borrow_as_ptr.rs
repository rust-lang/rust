use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_no_std_crate;
use clippy_utils::source::snippet_with_context;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, Mutability, Ty, TyKind};
use rustc_lint::LateContext;

use super::BORROW_AS_PTR;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    cast_expr: &'tcx Expr<'_>,
    cast_to: &'tcx Ty<'_>,
) {
    if matches!(cast_to.kind, TyKind::Ptr(_))
        && let ExprKind::AddrOf(BorrowKind::Ref, mutability, e) = cast_expr.kind
    {
        let core_or_std = if is_no_std_crate(cx) { "core" } else { "std" };
        let macro_name = match mutability {
            Mutability::Not => "addr_of",
            Mutability::Mut => "addr_of_mut",
        };
        let mut app = Applicability::MachineApplicable;
        let snip = snippet_with_context(cx, e.span, cast_expr.span.ctxt(), "..", &mut app).0;

        span_lint_and_sugg(
            cx,
            BORROW_AS_PTR,
            expr.span,
            "borrow as raw pointer",
            "try",
            format!("{}::ptr::{}!({})", core_or_std, macro_name, snip),
            Applicability::MachineApplicable,
        );
    }
}
