use clippy_utils::consts::constant_is_empty;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::{find_binding_init, path_to_local};
use rustc_hir::{Expr, HirId};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_span::sym;

use super::CONST_IS_EMPTY;

/// Expression whose initialization depend on a constant conditioned by a `#[cfg(…)]` directive will
/// not trigger the lint.
pub(super) fn check(cx: &LateContext<'_>, expr: &'_ Expr<'_>, receiver: &Expr<'_>) {
    if in_external_macro(cx.sess(), expr.span) || !receiver.span.eq_ctxt(expr.span) {
        return;
    }
    let init_expr = expr_or_init(cx, receiver);
    if !receiver.span.eq_ctxt(init_expr.span) {
        return;
    }
    if let Some(init_is_empty) = constant_is_empty(cx, init_expr) {
        span_lint(
            cx,
            CONST_IS_EMPTY,
            expr.span,
            &format!("this expression always evaluates to {init_is_empty:?}"),
        );
    }
}

fn is_under_cfg(cx: &LateContext<'_>, id: HirId) -> bool {
    cx.tcx
        .hir()
        .parent_id_iter(id)
        .any(|id| cx.tcx.hir().attrs(id).iter().any(|attr| attr.has_name(sym::cfg)))
}

/// Similar to [`clippy_utils::expr_or_init`], but does not go up the chain if the initialization
/// value depends on a `#[cfg(…)]` directive.
fn expr_or_init<'a, 'b, 'tcx: 'b>(cx: &LateContext<'tcx>, mut expr: &'a Expr<'b>) -> &'a Expr<'b> {
    while let Some(init) = path_to_local(expr)
        .and_then(|id| find_binding_init(cx, id))
        .filter(|init| cx.typeck_results().expr_adjustments(init).is_empty())
        .filter(|init| !is_under_cfg(cx, init.hir_id))
    {
        expr = init;
    }
    expr
}
