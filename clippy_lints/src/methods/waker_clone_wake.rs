use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{match_def_path, paths};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::sym;

use super::WAKER_CLONE_WAKE;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, recv: &'tcx Expr<'_>) {
    let ty = cx.typeck_results().expr_ty(recv);

    if let Some(did) = ty.ty_adt_def()
        && match_def_path(cx, did.did(), &paths::WAKER)
        && let ExprKind::MethodCall(func, waker_ref, &[], _) = recv.kind
        && func.ident.name == sym::clone
    {
        let mut applicability = Applicability::MachineApplicable;

        span_lint_and_sugg(
            cx,
            WAKER_CLONE_WAKE,
            expr.span,
            "cloning a `Waker` only to wake it",
            "replace with",
            format!(
                "{}.wake_by_ref()",
                snippet_with_applicability(cx, waker_ref.span, "..", &mut applicability)
            ),
            applicability,
        );
    }
}
