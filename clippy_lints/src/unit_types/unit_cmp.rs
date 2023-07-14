use clippy_utils::diagnostics::span_lint;
use clippy_utils::macros::{find_assert_eq_args, root_macro_call_first_node};
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;

use super::UNIT_CMP;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if expr.span.from_expansion() {
        if let Some(macro_call) = root_macro_call_first_node(cx, expr) {
            let macro_name = cx.tcx.item_name(macro_call.def_id);
            let result = match macro_name.as_str() {
                "assert_eq" | "debug_assert_eq" => "succeed",
                "assert_ne" | "debug_assert_ne" => "fail",
                _ => return,
            };
            let Some((left, _, _)) = find_assert_eq_args(cx, expr, macro_call.expn) else {
                return;
            };
            if !cx.typeck_results().expr_ty(left).is_unit() {
                return;
            }
            span_lint(
                cx,
                UNIT_CMP,
                macro_call.span,
                &format!("`{macro_name}` of unit values detected. This will always {result}"),
            );
        }
        return;
    }

    if let ExprKind::Binary(ref cmp, left, _) = expr.kind {
        let op = cmp.node;
        if op.is_comparison() && cx.typeck_results().expr_ty(left).is_unit() {
            let result = match op {
                BinOpKind::Eq | BinOpKind::Le | BinOpKind::Ge => "true",
                _ => "false",
            };
            span_lint(
                cx,
                UNIT_CMP,
                expr.span,
                &format!(
                    "{}-comparison of unit values detected. This will always be {result}",
                    op.as_str()
                ),
            );
        }
    }
}
