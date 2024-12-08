use clippy_utils::diagnostics::span_lint;
use rustc_ast::ast::{Expr, ExprKind, UnOp};
use rustc_lint::EarlyContext;

use super::DOUBLE_NEG;

pub(super) fn check(cx: &EarlyContext<'_>, expr: &Expr) {
    if let ExprKind::Unary(UnOp::Neg, ref inner) = expr.kind {
        if let ExprKind::Unary(UnOp::Neg, _) = inner.kind {
            span_lint(
                cx,
                DOUBLE_NEG,
                expr.span,
                "`--x` could be misinterpreted as pre-decrement by C programmers, is usually a no-op",
            );
        }
    }
}
