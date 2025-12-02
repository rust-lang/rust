use clippy_utils::diagnostics::span_lint;
use clippy_utils::{SpanlessEq, eq_expr_value, hash_expr, search_same};
use rustc_hir::Expr;
use rustc_lint::LateContext;

use super::SAME_FUNCTIONS_IN_IF_CONDITION;

/// Implementation of `SAME_FUNCTIONS_IN_IF_CONDITION`.
pub(super) fn check(cx: &LateContext<'_>, conds: &[&Expr<'_>]) {
    let eq: &dyn Fn(&&Expr<'_>, &&Expr<'_>) -> bool = &|&lhs, &rhs| -> bool {
        // Do not lint if any expr originates from a macro
        if lhs.span.from_expansion() || rhs.span.from_expansion() {
            return false;
        }
        // Do not spawn warning if `IFS_SAME_COND` already produced it.
        if eq_expr_value(cx, lhs, rhs) {
            return false;
        }
        SpanlessEq::new(cx).eq_expr(lhs, rhs)
    };

    for group in search_same(conds, |e| hash_expr(cx, e), eq) {
        let spans: Vec<_> = group.into_iter().map(|expr| expr.span).collect();
        span_lint(
            cx,
            SAME_FUNCTIONS_IN_IF_CONDITION,
            spans,
            "these `if` branches have the same function call",
        );
    }
}
