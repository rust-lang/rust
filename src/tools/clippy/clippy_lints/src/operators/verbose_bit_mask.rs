use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg::Sugg;
use rustc_ast::ast::LitKind;
use rustc_data_structures::packed::Pu128;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;

use super::VERBOSE_BIT_MASK;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    op: BinOpKind,
    left: &'tcx Expr<'_>,
    right: &'tcx Expr<'_>,
    threshold: u64,
) {
    if BinOpKind::Eq == op
        && let ExprKind::Binary(op1, left1, right1) = &left.kind
        && BinOpKind::BitAnd == op1.node
        && let ExprKind::Lit(lit) = &right1.kind
        && let LitKind::Int(Pu128(n), _) = lit.node
        && let ExprKind::Lit(lit1) = &right.kind
        && let LitKind::Int(Pu128(0), _) = lit1.node
        && n.leading_zeros() == n.count_zeros()
        && n > u128::from(threshold)
    {
        span_lint_and_then(
            cx,
            VERBOSE_BIT_MASK,
            e.span,
            "bit mask could be simplified with a call to `trailing_zeros`",
            |diag| {
                let sugg = Sugg::hir(cx, left1, "...").maybe_paren();
                diag.span_suggestion(
                    e.span,
                    "try",
                    format!("{sugg}.trailing_zeros() >= {}", n.count_ones()),
                    Applicability::MaybeIncorrect,
                );
            },
        );
    }
}
