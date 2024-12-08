use clippy_utils::SpanlessEq;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::ty::is_type_lang_item;
use rustc_ast::LitKind;
use rustc_hir::{ExprKind, LangItem};
use rustc_lint::LateContext;

use super::NO_EFFECT_REPLACE;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx rustc_hir::Expr<'_>,
    arg1: &'tcx rustc_hir::Expr<'_>,
    arg2: &'tcx rustc_hir::Expr<'_>,
) {
    let ty = cx.typeck_results().expr_ty(expr).peel_refs();
    if !(ty.is_str() || is_type_lang_item(cx, ty, LangItem::String)) {
        return;
    }

    if let ExprKind::Lit(spanned) = &arg1.kind
        && let Some(param1) = lit_string_value(&spanned.node)
        && let ExprKind::Lit(spanned) = &arg2.kind
        && let LitKind::Str(param2, _) = &spanned.node
        && param1 == param2.as_str()
    {
        span_lint(cx, NO_EFFECT_REPLACE, expr.span, "replacing text with itself");
        return;
    }

    if SpanlessEq::new(cx).eq_expr(arg1, arg2) {
        span_lint(cx, NO_EFFECT_REPLACE, expr.span, "replacing text with itself");
    }
}

fn lit_string_value(node: &LitKind) -> Option<String> {
    match node {
        LitKind::Char(value) => Some(value.to_string()),
        LitKind::Str(value, _) => Some(value.as_str().to_owned()),
        _ => None,
    }
}
