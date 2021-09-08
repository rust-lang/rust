use super::WHILE_LET_LOOP;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher;
use clippy_utils::source::snippet_with_applicability;
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, ExprKind, MatchSource, Pat, StmtKind};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::lint::in_external_macro;

pub(super) fn check(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, loop_block: &'tcx Block<'_>) {
    // extract the expression from the first statement (if any) in a block
    let inner_stmt_expr = extract_expr_from_first_stmt(loop_block);
    // or extract the first expression (if any) from the block
    if let Some(inner) = inner_stmt_expr.or_else(|| extract_first_expr(loop_block)) {
        if let Some(higher::IfLet {
            let_pat,
            let_expr,
            if_else: Some(if_else),
            ..
        }) = higher::IfLet::hir(cx, inner)
        {
            if is_simple_break_expr(if_else) {
                could_be_while_let(cx, expr, let_pat, let_expr);
            }
        }

        if let ExprKind::Match(matchexpr, arms, MatchSource::Normal) = inner.kind {
            if arms.len() == 2
                && arms[0].guard.is_none()
                && arms[1].guard.is_none()
                && is_simple_break_expr(arms[1].body)
            {
                could_be_while_let(cx, expr, arms[0].pat, matchexpr);
            }
        }
    }
}

/// If a block begins with a statement (possibly a `let` binding) and has an
/// expression, return it.
fn extract_expr_from_first_stmt<'tcx>(block: &Block<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    if let Some(first_stmt) = block.stmts.get(0) {
        if let StmtKind::Local(local) = first_stmt.kind {
            return local.init;
        }
    }
    None
}

/// If a block begins with an expression (with or without semicolon), return it.
fn extract_first_expr<'tcx>(block: &Block<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    match block.expr {
        Some(expr) if block.stmts.is_empty() => Some(expr),
        None if !block.stmts.is_empty() => match block.stmts[0].kind {
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => Some(expr),
            StmtKind::Local(..) | StmtKind::Item(..) => None,
        },
        _ => None,
    }
}

/// Returns `true` if expr contains a single break expr without destination label
/// and
/// passed expression. The expression may be within a block.
fn is_simple_break_expr(expr: &Expr<'_>) -> bool {
    match expr.kind {
        ExprKind::Break(dest, ref passed_expr) if dest.label.is_none() && passed_expr.is_none() => true,
        ExprKind::Block(b, _) => extract_first_expr(b).map_or(false, |subexpr| is_simple_break_expr(subexpr)),
        _ => false,
    }
}

fn could_be_while_let<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    let_pat: &'tcx Pat<'_>,
    let_expr: &'tcx Expr<'_>,
) {
    if in_external_macro(cx.sess(), expr.span) {
        return;
    }

    // NOTE: we used to build a body here instead of using
    // ellipsis, this was removed because:
    // 1) it was ugly with big bodies;
    // 2) it was not indented properly;
    // 3) it wasn’t very smart (see #675).
    let mut applicability = Applicability::HasPlaceholders;
    span_lint_and_sugg(
        cx,
        WHILE_LET_LOOP,
        expr.span,
        "this loop could be written as a `while let` loop",
        "try",
        format!(
            "while let {} = {} {{ .. }}",
            snippet_with_applicability(cx, let_pat.span, "..", &mut applicability),
            snippet_with_applicability(cx, let_expr.span, "..", &mut applicability),
        ),
        applicability,
    );
}
