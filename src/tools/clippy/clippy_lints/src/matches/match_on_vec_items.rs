use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::ty::{is_type_diagnostic_item, is_type_lang_item};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LangItem};
use rustc_lint::LateContext;
use rustc_span::sym;

use super::MATCH_ON_VEC_ITEMS;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, scrutinee: &'tcx Expr<'_>) {
    if let Some(idx_expr) = is_vec_indexing(cx, scrutinee)
        && let ExprKind::Index(vec, idx, _) = idx_expr.kind
    {
        // FIXME: could be improved to suggest surrounding every pattern with Some(_),
        // but only when `or_patterns` are stabilized.
        span_lint_and_sugg(
            cx,
            MATCH_ON_VEC_ITEMS,
            scrutinee.span,
            "indexing into a vector may panic",
            "try",
            format!("{}.get({})", snippet(cx, vec.span, ".."), snippet(cx, idx.span, "..")),
            Applicability::MaybeIncorrect,
        );
    }
}

fn is_vec_indexing<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    if let ExprKind::Index(array, index, _) = expr.kind
        && is_vector(cx, array)
        && !is_full_range(cx, index)
    {
        return Some(expr);
    }

    None
}

fn is_vector(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let ty = cx.typeck_results().expr_ty(expr);
    let ty = ty.peel_refs();
    is_type_diagnostic_item(cx, ty, sym::Vec)
}

fn is_full_range(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let ty = cx.typeck_results().expr_ty(expr);
    let ty = ty.peel_refs();
    is_type_lang_item(cx, ty, LangItem::RangeFull)
}
