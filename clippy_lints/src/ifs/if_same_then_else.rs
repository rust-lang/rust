use clippy_utils::SpanlessEq;
use clippy_utils::diagnostics::span_lint_and_note;
use clippy_utils::higher::has_let_expr;
use rustc_hir::{Block, Expr};
use rustc_lint::LateContext;

use super::IF_SAME_THEN_ELSE;

pub(super) fn check(cx: &LateContext<'_>, conds: &[&Expr<'_>], blocks: &[&Block<'_>]) -> bool {
    let mut eq = SpanlessEq::new(cx);
    blocks
        .array_windows::<2>()
        .enumerate()
        .fold(true, |all_eq, (i, &[lhs, rhs])| {
            if eq.eq_block(lhs, rhs) && !has_let_expr(conds[i]) && conds.get(i + 1).is_none_or(|e| !has_let_expr(e)) {
                span_lint_and_note(
                    cx,
                    IF_SAME_THEN_ELSE,
                    lhs.span,
                    "this `if` has identical blocks",
                    Some(rhs.span),
                    "same as this",
                );
                all_eq
            } else {
                false
            }
        })
}
