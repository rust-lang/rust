use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_range_full;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;
use rustc_span::Span;

use super::ITER_WITH_DRAIN;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, span: Span, arg: &Expr<'_>) {
    if !matches!(recv.kind, ExprKind::Field(..))
        && let Some(adt) = cx.typeck_results().expr_ty(recv).ty_adt_def()
        && let Some(ty_name) = cx.tcx.get_diagnostic_name(adt.did())
        && matches!(ty_name, sym::Vec | sym::VecDeque)
        && let ExprKind::Path(QPath::Resolved(None, container_path)) = recv.kind
        && is_range_full(cx, arg, Some(container_path))
    {
        span_lint_and_sugg(
            cx,
            ITER_WITH_DRAIN,
            span.with_hi(expr.span.hi()),
            &format!("`drain(..)` used on a `{ty_name}`"),
            "try",
            "into_iter()".to_string(),
            Applicability::MaybeIncorrect,
        );
    };
}
