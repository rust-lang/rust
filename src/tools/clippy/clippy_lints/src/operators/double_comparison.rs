use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::eq_expr_value;
use clippy_utils::source::snippet_with_applicability;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::Span;

use super::DOUBLE_COMPARISONS;

pub(super) fn check(cx: &LateContext<'_>, op: BinOpKind, lhs: &Expr<'_>, rhs: &Expr<'_>, span: Span) {
    if let ExprKind::Binary(lop, llhs, lrhs) = lhs.kind
        && let ExprKind::Binary(rop, rlhs, rrhs) = rhs.kind
        && eq_expr_value(cx, llhs, rlhs)
        && eq_expr_value(cx, lrhs, rrhs)
    {
        let op = match (op, lop.node, rop.node) {
            // x == y || x < y => x <= y
            (BinOpKind::Or, BinOpKind::Eq, BinOpKind::Lt)
            // x < y || x == y => x <= y
            | (BinOpKind::Or, BinOpKind::Lt, BinOpKind::Eq) => {
                "<="
            },
            // x == y || x > y => x >= y
            (BinOpKind::Or, BinOpKind::Eq, BinOpKind::Gt)
            // x > y || x == y => x >= y
            | (BinOpKind::Or, BinOpKind::Gt, BinOpKind::Eq) => {
                ">="
            },
            // x < y || x > y => x != y
            (BinOpKind::Or, BinOpKind::Lt, BinOpKind::Gt)
            // x > y || x < y => x != y
            | (BinOpKind::Or, BinOpKind::Gt, BinOpKind::Lt) => {
                "!="
            },
            // x <= y && x >= y => x == y
            (BinOpKind::And, BinOpKind::Le, BinOpKind::Ge)
            // x >= y && x <= y => x == y
            | (BinOpKind::And, BinOpKind::Ge, BinOpKind::Le) => {
                "=="
            },
            _ => return,
        };

        let mut applicability = Applicability::MachineApplicable;
        let lhs_str = snippet_with_applicability(cx, llhs.span, "", &mut applicability);
        let rhs_str = snippet_with_applicability(cx, lrhs.span, "", &mut applicability);
        let sugg = format!("{lhs_str} {op} {rhs_str}");
        span_lint_and_sugg(
            cx,
            DOUBLE_COMPARISONS,
            span,
            "this binary expression can be simplified",
            "try",
            sugg,
            applicability,
        );
    }
}
