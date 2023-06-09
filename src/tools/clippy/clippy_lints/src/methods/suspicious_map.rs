use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::usage::mutated_variables;
use clippy_utils::{expr_or_init, is_trait_method};
use if_chain::if_chain;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::SUSPICIOUS_MAP;

pub fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, count_recv: &hir::Expr<'_>, map_arg: &hir::Expr<'_>) {
    if_chain! {
        if is_trait_method(cx, count_recv, sym::Iterator);
        if let hir::ExprKind::Closure(closure) = expr_or_init(cx, map_arg).kind;
        let closure_body = cx.tcx.hir().body(closure.body);
        if !cx.typeck_results().expr_ty(closure_body.value).is_unit();
        then {
            if let Some(map_mutated_vars) = mutated_variables(closure_body.value, cx) {
                // A variable is used mutably inside of the closure. Suppress the lint.
                if !map_mutated_vars.is_empty() {
                    return;
                }
            }
            span_lint_and_help(
                cx,
                SUSPICIOUS_MAP,
                expr.span,
                "this call to `map()` won't have an effect on the call to `count()`",
                None,
                "make sure you did not confuse `map` with `filter`, `for_each` or `inspect`",
            );
        }
    }
}
