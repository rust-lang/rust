use clippy_utils::diagnostics::span_lint_and_note;
use clippy_utils::expr_or_init;
use rustc_ast::LitKind;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_span::source_map::Spanned;

use super::CONST_IS_EMPTY;

pub(super) fn check(cx: &LateContext<'_>, expr: &'_ Expr<'_>, receiver: &Expr<'_>) {
    if in_external_macro(cx.sess(), expr.span) || !receiver.span.eq_ctxt(expr.span) {
        return;
    }
    let init_expr = expr_or_init(cx, receiver);
    if let Some(init_is_empty) = is_empty(init_expr)
        && init_expr.span.eq_ctxt(receiver.span)
    {
        span_lint_and_note(
            cx,
            CONST_IS_EMPTY,
            expr.span,
            &format!("this expression always evaluates to {init_is_empty:?}"),
            Some(init_expr.span),
            "because its initialization value is constant",
        );
    }
}

fn is_empty(expr: &'_ rustc_hir::Expr<'_>) -> Option<bool> {
    if let ExprKind::Lit(Spanned { node, .. }) = expr.kind {
        match node {
            LitKind::Str(sym, _) => Some(sym.is_empty()),
            LitKind::ByteStr(value, _) | LitKind::CStr(value, _) => Some(value.is_empty()),
            _ => None,
        }
    } else {
        None
    }
}
