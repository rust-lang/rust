use super::FORMAT_COLLECT;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::{is_format_macro, root_macro_call_first_node};
use clippy_utils::ty::is_type_lang_item;
use rustc_hir::{Expr, ExprKind, LangItem};
use rustc_lint::LateContext;
use rustc_span::Span;

/// Same as `peel_blocks` but only actually considers blocks that are not from an expansion.
/// This is needed because always calling `peel_blocks` would otherwise remove parts of the
/// `format!` macro, which would cause `root_macro_call_first_node` to return `None`.
fn peel_non_expn_blocks<'tcx>(expr: &'tcx Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    match expr.kind {
        ExprKind::Block(block, _) if !expr.span.from_expansion() => peel_non_expn_blocks(block.expr?),
        _ => Some(expr),
    }
}

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, map_arg: &Expr<'_>, map_span: Span) {
    if is_type_lang_item(cx, cx.typeck_results().expr_ty(expr), LangItem::String)
        && let ExprKind::Closure(closure) = map_arg.kind
        && let body = cx.tcx.hir_body(closure.body)
        && let Some(value) = peel_non_expn_blocks(body.value)
        && let Some(mac) = root_macro_call_first_node(cx, value)
        && is_format_macro(cx, mac.def_id)
    {
        span_lint_and_then(
            cx,
            FORMAT_COLLECT,
            expr.span,
            "use of `format!` to build up a string from an iterator",
            |diag| {
                diag.span_help(map_span, "call `fold` instead")
                    .span_help(value.span.source_callsite(), "... and use the `write!` macro here")
                    .note("this can be written more efficiently by appending to a `String` directly");
            },
        );
    }
}
