use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_lang_item;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LangItem, is_range_literal};
use rustc_lint::LateContext;

use super::SLICED_STRING_AS_BYTES;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>) {
    if let ExprKind::Index(indexed, index, _) = recv.kind
        && is_range_literal(index)
        && let ty = cx.typeck_results().expr_ty(indexed).peel_refs()
        && (ty.is_str() || is_type_lang_item(cx, ty, LangItem::String))
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
