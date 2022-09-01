use super::WHILE_LET_LOOP;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::needs_ordered_drop;
use clippy_utils::visitors::any_temporaries_need_ordered_drop;
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, ExprKind, Local, MatchSource, Pat, StmtKind};
use rustc_lint::LateContext;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, loop_block: &'tcx Block<'_>) {
    let (init, has_trailing_exprs) = match (loop_block.stmts, loop_block.expr) {
        ([stmt, stmts @ ..], expr) => {
            if let StmtKind::Local(&Local {
                init: Some(e),
                els: None,
                ..
            })
            | StmtKind::Semi(e)
            | StmtKind::Expr(e) = stmt.kind
            {
                (e, !stmts.is_empty() || expr.is_some())
            } else {
                return;
            }
        },
        ([], Some(e)) => (e, false),
        _ => return,
    };

    if let Some(if_let) = higher::IfLet::hir(cx, init)
        && let Some(else_expr) = if_let.if_else
        && is_simple_break_expr(else_expr)
    {
        could_be_while_let(cx, expr, if_let.let_pat, if_let.let_expr, has_trailing_exprs);
    } else if let ExprKind::Match(scrutinee, [arm1, arm2], MatchSource::Normal) = init.kind
        && arm1.guard.is_none()
        && arm2.guard.is_none()
        && is_simple_break_expr(arm2.body)
    {
        could_be_while_let(cx, expr, arm1.pat, scrutinee, has_trailing_exprs);
    }
}

/// Returns `true` if expr contains a single break expression without a label or eub-expression.
fn is_simple_break_expr(e: &Expr<'_>) -> bool {
    matches!(peel_blocks(e).kind, ExprKind::Break(dest, None) if dest.label.is_none())
}

/// Removes any blocks containing only a single expression.
fn peel_blocks<'tcx>(e: &'tcx Expr<'tcx>) -> &'tcx Expr<'tcx> {
    if let ExprKind::Block(b, _) = e.kind {
        match (b.stmts, b.expr) {
            ([s], None) => {
                if let StmtKind::Expr(e) | StmtKind::Semi(e) = s.kind {
                    peel_blocks(e)
                } else {
                    e
                }
            },
            ([], Some(e)) => peel_blocks(e),
            _ => e,
        }
    } else {
        e
    }
}

fn could_be_while_let<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    let_pat: &'tcx Pat<'_>,
    let_expr: &'tcx Expr<'_>,
    has_trailing_exprs: bool,
) {
    if has_trailing_exprs
        && (needs_ordered_drop(cx, cx.typeck_results().expr_ty(let_expr))
            || any_temporaries_need_ordered_drop(cx, let_expr))
    {
        // Switching to a `while let` loop will extend the lifetime of some values.
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
