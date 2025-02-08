use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::{SpanlessEq, higher, is_integer_const, is_trait_method};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::LateContext;
use rustc_span::sym;

use super::RANGE_ZIP_WITH_LEN;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, recv: &'tcx Expr<'_>, zip_arg: &'tcx Expr<'_>) {
    if is_trait_method(cx, expr, sym::Iterator)
        // range expression in `.zip()` call: `0..x.len()`
        && let Some(higher::Range { start: Some(start), end: Some(end), .. }) = higher::Range::hir(zip_arg)
        && is_integer_const(cx, start, 0)
        // `.len()` call
        && let ExprKind::MethodCall(len_path, len_recv, [], _) = end.kind
        && len_path.ident.name == sym::len
        // `.iter()` and `.len()` called on same `Path`
        && let ExprKind::Path(QPath::Resolved(_, iter_path)) = recv.kind
        && let ExprKind::Path(QPath::Resolved(_, len_path)) = len_recv.kind
        && SpanlessEq::new(cx).eq_path_segments(iter_path.segments, len_path.segments)
    {
        span_lint_and_sugg(
            cx,
            RANGE_ZIP_WITH_LEN,
            expr.span,
            "using `.zip()` with a range and `.len()`",
            "try",
            format!("{}.iter().enumerate()", snippet(cx, recv.span, "_")),
            Applicability::MachineApplicable,
        );
    }
}
