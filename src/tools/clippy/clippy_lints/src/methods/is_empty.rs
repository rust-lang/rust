use clippy_utils::consts::ConstEvalCtxt;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::macros::{is_assert_macro, root_macro_call};
use clippy_utils::{find_binding_init, get_parent_expr, is_inside_always_const_context, path_to_local};
use rustc_hir::{Expr, HirId};
use rustc_lint::{LateContext, LintContext};
use rustc_span::sym;

use super::CONST_IS_EMPTY;

/// Expression whose initialization depend on a constant conditioned by a `#[cfg(…)]` directive will
/// not trigger the lint.
pub(super) fn check(cx: &LateContext<'_>, expr: &'_ Expr<'_>, receiver: &Expr<'_>) {
    if expr.span.in_external_macro(cx.sess().source_map()) || !receiver.span.eq_ctxt(expr.span) {
        return;
    }
    if let Some(parent) = get_parent_expr(cx, expr)
        && let Some(parent) = get_parent_expr(cx, parent)
        && is_inside_always_const_context(cx.tcx, expr.hir_id)
        && let Some(macro_call) = root_macro_call(parent.span)
        && is_assert_macro(cx, macro_call.def_id)
    {
        return;
    }
    let init_expr = expr_or_init(cx, receiver);
    if !receiver.span.eq_ctxt(init_expr.span) {
        return;
    }
    if let Some(init_is_empty) = ConstEvalCtxt::new(cx).eval_is_empty(init_expr) {
        span_lint(
            cx,
            CONST_IS_EMPTY,
            expr.span,
            format!("this expression always evaluates to {init_is_empty:?}"),
        );
    }
}

fn is_under_cfg(cx: &LateContext<'_>, id: HirId) -> bool {
    cx.tcx
        .hir_parent_id_iter(id)
        .any(|id| cx.tcx.hir_attrs(id).iter().any(|attr| attr.has_name(sym::cfg_trace)))
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
