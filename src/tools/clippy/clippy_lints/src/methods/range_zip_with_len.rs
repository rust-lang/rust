use clippy_utils::diagnostics::span_lint;
use clippy_utils::source::snippet;
use clippy_utils::{higher, is_integer_const, is_trait_method, SpanlessEq};
use if_chain::if_chain;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::LateContext;
use rustc_span::sym;

use super::RANGE_ZIP_WITH_LEN;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, recv: &'tcx Expr<'_>, zip_arg: &'tcx Expr<'_>) {
    if_chain! {
        if is_trait_method(cx, expr, sym::Iterator);
        // range expression in `.zip()` call: `0..x.len()`
        if let Some(higher::Range { start: Some(start), end: Some(end), .. }) = higher::Range::hir(zip_arg);
        if is_integer_const(cx, start, 0);
        // `.len()` call
        if let ExprKind::MethodCall(len_path, len_recv, [], _) = end.kind;
        if len_path.ident.name == sym::len;
        // `.iter()` and `.len()` called on same `Path`
        if let ExprKind::Path(QPath::Resolved(_, iter_path)) = recv.kind;
        if let ExprKind::Path(QPath::Resolved(_, len_path)) = len_recv.kind;
        if SpanlessEq::new(cx).eq_path_segments(iter_path.segments, len_path.segments);
        then {
            span_lint(cx,
                RANGE_ZIP_WITH_LEN,
                expr.span,
                &format!("it is more idiomatic to use `{}.iter().enumerate()`",
                    snippet(cx, recv.span, "_"))
            );
        }
    }
}
