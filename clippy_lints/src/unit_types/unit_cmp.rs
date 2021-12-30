use clippy_utils::diagnostics::span_lint;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::hygiene::{ExpnKind, MacroKind};

use super::UNIT_CMP;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if expr.span.from_expansion() {
        if let Some(callee) = expr.span.source_callee() {
            if let ExpnKind::Macro(MacroKind::Bang, symbol) = callee.kind {
                if let ExprKind::Binary(ref cmp, left, _) = expr.kind {
                    let op = cmp.node;
                    if op.is_comparison() && cx.typeck_results().expr_ty(left).is_unit() {
                        let result = match symbol.as_str() {
                            "assert_eq" | "debug_assert_eq" => "succeed",
                            "assert_ne" | "debug_assert_ne" => "fail",
                            _ => return,
                        };
                        span_lint(
                            cx,
                            UNIT_CMP,
                            expr.span,
                            &format!(
                                "`{}` of unit values detected. This will always {}",
                                symbol.as_str(),
                                result
                            ),
                        );
                    }
                }
            }
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
                    "{}-comparison of unit values detected. This will always be {}",
                    op.as_str(),
                    result
                ),
            );
        }
    }
}
