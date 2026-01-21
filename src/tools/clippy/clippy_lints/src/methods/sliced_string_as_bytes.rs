use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher;
use clippy_utils::res::MaybeDef;
use clippy_utils::source::snippet_with_applicability;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LangItem};
use rustc_lint::LateContext;

use super::SLICED_STRING_AS_BYTES;

/// Checks if `index` is any type of range except `RangeFull` (i.e. `..`)
fn is_bounded_range_literal(cx: &LateContext<'_>, index: &Expr<'_>) -> bool {
    higher::Range::hir(cx, index).is_some_and(|range| Option::or(range.start, range.end).is_some())
}

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>) {
    if let ExprKind::Index(indexed, index, _) = recv.kind
        && is_bounded_range_literal(cx, index)
        && let ty = cx.typeck_results().expr_ty(indexed).peel_refs()
        && (ty.is_str() || ty.is_lang_item(cx, LangItem::String))
    {
        let mut applicability = Applicability::MaybeIncorrect;
        let stringish = snippet_with_applicability(cx, indexed.span, "_", &mut applicability);
        let range = snippet_with_applicability(cx, index.span, "_", &mut applicability);
        span_lint_and_sugg(
            cx,
            SLICED_STRING_AS_BYTES,
            expr.span,
            "calling `as_bytes` after slicing a string",
            "try",
            format!("&{stringish}.as_bytes()[{range}]"),
            applicability,
        );
    }
}
