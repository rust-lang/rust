use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::res::MaybeResPath as _;
use clippy_utils::source::snippet_with_context;
use clippy_utils::sugg::has_enclosing_paren;
use clippy_utils::visitors::for_each_expr;
use clippy_utils::{binary_expr_needs_parentheses, fn_def_id, span_contains_non_whitespace};
use core::ops::ControlFlow;
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, PatKind, Stmt, StmtKind};
use rustc_lint::{LateContext, Level, LintContext as _};
use rustc_middle::ty::GenericArgKind;
use rustc_span::edition::Edition;

use super::LET_AND_RETURN;

pub(super) fn check_block<'tcx>(cx: &LateContext<'tcx>, block: &'tcx Block<'_>) {
    // we need both a let-binding stmt and an expr
    if let Some(retexpr) = block.expr
        && let Some(stmt) = block.stmts.last()
        && let StmtKind::Let(local) = &stmt.kind
        && local.ty.is_none()
        && local.els.is_none()
        && cx.tcx.hir_attrs(local.hir_id).is_empty()
        && let Some(initexpr) = &local.init
        && let PatKind::Binding(_, local_id, _, _) = local.pat.kind
        && retexpr.res_local_id() == Some(local_id)
        && (cx.sess().edition() >= Edition::Edition2024 || !last_statement_borrows(cx, initexpr))
        && !initexpr.span.in_external_macro(cx.sess().source_map())
        && !retexpr.span.in_external_macro(cx.sess().source_map())
        && !local.span.from_expansion()
        && has_lint_attrs_or_only_whitespace_between(cx, retexpr, stmt)
    {
        span_lint_hir_and_then(
            cx,
            LET_AND_RETURN,
            retexpr.hir_id,
            retexpr.span,
            "returning the result of a `let` binding from a block",
            |err| {
                err.span_label(local.span, "unnecessary `let` binding");

                let mut app = Applicability::MachineApplicable;
                let (src, _) = snippet_with_context(cx, initexpr.span, local.span.ctxt(), "..", &mut app);

                let sugg = if binary_expr_needs_parentheses(initexpr) {
                    if has_enclosing_paren(&src) {
                        src.to_string()
                    } else {
                        format!("({src})")
                    }
                } else if !cx.typeck_results().expr_adjustments(retexpr).is_empty() {
                    if has_enclosing_paren(&src) {
                        format!("{src} as _")
                    } else {
                        format!("({src}) as _")
                    }
                } else {
                    src.to_string()
                };
                err.multipart_suggestion(
                    "return the expression directly",
                    vec![(local.span, String::new()), (retexpr.span, sugg)],
                    app,
                );
            },
        );
    }
}
fn last_statement_borrows<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    for_each_expr(cx.tcx, expr, |e| {
        if let Some(def_id) = fn_def_id(cx, e)
            && cx
                .tcx
                .fn_sig(def_id)
                .instantiate_identity()
                .skip_norm_wip()
                .skip_binder()
                .output()
                .walk()
                .any(|arg| matches!(arg.kind(), GenericArgKind::Lifetime(re) if !re.is_static()))
        {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .is_some()
}

/// Returns true if the return expression has lint-level attributes,
/// or if there is only whitespace between `let` and return expression.
/// Non-lint attrs like `#[cfg]` should still block.
fn has_lint_attrs_or_only_whitespace_between(cx: &LateContext<'_>, retexpr: &Expr<'_>, stmt: &Stmt<'_>) -> bool {
    // TODO: Turn into find_attr! when lint level attr parsing is done.
    let retexpr_attrs = cx.tcx.hir_attrs(retexpr.hir_id);

    retexpr_attrs
        .iter()
        .any(|a| a.name().is_some_and(|name| Level::from_symbol(name).is_some()))
        || !span_contains_non_whitespace(cx, stmt.span.between(retexpr.span), false)
}
