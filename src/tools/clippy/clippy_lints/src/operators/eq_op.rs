use clippy_utils::ast_utils::is_useless_with_eq_exprs;
use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::macros::{find_assert_eq_args, first_node_macro_backtrace};
use clippy_utils::{eq_expr_value, is_in_test_function};
use rustc_hir::{BinOpKind, Expr};
use rustc_lint::LateContext;

use super::EQ_OP;

pub(crate) fn check_assert<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
    if let Some((macro_call, macro_name))
        = first_node_macro_backtrace(cx, e).find_map(|macro_call| {
            let name = cx.tcx.item_name(macro_call.def_id);
            matches!(name.as_str(), "assert_eq" | "assert_ne" | "debug_assert_eq" | "debug_assert_ne")
                .then(|| (macro_call, name))
        })
        && let Some((lhs, rhs, _)) = find_assert_eq_args(cx, e, macro_call.expn)
        && eq_expr_value(cx, lhs, rhs)
        && macro_call.is_local()
        && !is_in_test_function(cx.tcx, e.hir_id)
    {
        span_lint(
            cx,
            EQ_OP,
            lhs.span.to(rhs.span),
            &format!("identical args used in this `{macro_name}!` macro call"),
        );
    }
}

pub(crate) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    op: BinOpKind,
    left: &'tcx Expr<'_>,
    right: &'tcx Expr<'_>,
) {
    if is_useless_with_eq_exprs(op.into()) && eq_expr_value(cx, left, right) && !is_in_test_function(cx.tcx, e.hir_id) {
        span_lint_and_then(
            cx,
            EQ_OP,
            e.span,
            &format!("equal expressions as operands to `{}`", op.as_str()),
            |diag| {
                if let BinOpKind::Ne = op && cx.typeck_results().expr_ty(left).is_floating_point() {
                    diag.note("if you intended to check if the operand is NaN, use `.is_nan()` instead");
                }
            },
        );
    }
}
