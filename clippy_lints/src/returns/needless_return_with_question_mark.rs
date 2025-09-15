use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::res::MaybeQPath;
use clippy_utils::{is_from_proc_macro, is_inside_let_else, is_res_lang_ctor};
use rustc_errors::Applicability;
use rustc_hir::LangItem::ResultErr;
use rustc_hir::{ExprKind, HirId, ItemKind, MatchSource, Node, OwnerNode, Stmt, StmtKind};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::ty::adjustment::Adjust;

use super::NEEDLESS_RETURN_WITH_QUESTION_MARK;

pub(super) fn check_stmt<'tcx>(cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
    if !stmt.span.in_external_macro(cx.sess().source_map())
        && let StmtKind::Semi(expr) = stmt.kind
        && let ExprKind::Ret(Some(ret)) = expr.kind
        // return Err(...)? desugars to a match
        // over a Err(...).branch()
        // which breaks down to a branch call, with the callee being
        // the constructor of the Err variant
        && let ExprKind::Match(maybe_cons, _, MatchSource::TryDesugar(_)) = ret.kind
        && let ExprKind::Call(_, [maybe_result_err]) = maybe_cons.kind
        && let ExprKind::Call(maybe_constr, _) = maybe_result_err.kind
        && is_res_lang_ctor(cx, maybe_constr.res(cx), ResultErr)

        // Ensure this is not the final stmt, otherwise removing it would cause a compile error
        && let OwnerNode::Item(item) = cx.tcx.hir_owner_node(cx.tcx.hir_get_parent_item(expr.hir_id))
        && let ItemKind::Fn { body, .. } = item.kind
        && let block = cx.tcx.hir_body(body).value
        && let ExprKind::Block(block, _) = block.kind
        && !is_inside_let_else(cx.tcx, expr)
        && let [.., final_stmt] = block.stmts
        && final_stmt.hir_id != stmt.hir_id
        && !is_from_proc_macro(cx, expr)
        && !stmt_needs_never_type(cx, stmt.hir_id)
    {
        span_lint_and_sugg(
            cx,
            NEEDLESS_RETURN_WITH_QUESTION_MARK,
            expr.span.until(ret.span),
            "unneeded `return` statement with `?` operator",
            "remove it",
            String::new(),
            Applicability::MachineApplicable,
        );
    }
}

/// Checks if a return statement is "needed" in the middle of a block, or if it can be removed.
/// This is the case when the enclosing block expression is coerced to some other type,
/// which only works because of the never-ness of `return` expressions
fn stmt_needs_never_type(cx: &LateContext<'_>, stmt_hir_id: HirId) -> bool {
    cx.tcx
        .hir_parent_iter(stmt_hir_id)
        .find_map(|(_, node)| if let Node::Expr(expr) = node { Some(expr) } else { None })
        .is_some_and(|e| {
            cx.typeck_results()
                .expr_adjustments(e)
                .iter()
                .any(|adjust| adjust.target != cx.tcx.types.unit && matches!(adjust.kind, Adjust::NeverToAny))
        })
}
