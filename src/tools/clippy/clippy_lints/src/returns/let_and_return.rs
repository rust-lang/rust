use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::res::MaybeResPath;
use clippy_utils::source::SpanRangeExt;
use clippy_utils::sugg::has_enclosing_paren;
use clippy_utils::visitors::for_each_expr;
use clippy_utils::{binary_expr_needs_parentheses, fn_def_id, span_contains_cfg};
use core::ops::ControlFlow;
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, PatKind, StmtKind};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::ty::GenericArgKind;
use rustc_span::edition::Edition;

use super::LET_AND_RETURN;

pub(super) fn check_block<'tcx>(cx: &LateContext<'tcx>, block: &'tcx Block<'_>) {
    // we need both a let-binding stmt and an expr
    if let Some(retexpr) = block.expr
        && let Some(stmt) = block.stmts.last()
        && let StmtKind::Let(local) = &stmt.kind
        && local.ty.is_none()
        && cx.tcx.hir_attrs(local.hir_id).is_empty()
        && let Some(initexpr) = &local.init
        && let PatKind::Binding(_, local_id, _, _) = local.pat.kind
        && retexpr.res_local_id() == Some(local_id)
        && (cx.sess().edition() >= Edition::Edition2024 || !last_statement_borrows(cx, initexpr))
        && !initexpr.span.in_external_macro(cx.sess().source_map())
        && !retexpr.span.in_external_macro(cx.sess().source_map())
        && !local.span.from_expansion()
        && !span_contains_cfg(cx, stmt.span.between(retexpr.span))
    {
        span_lint_hir_and_then(
            cx,
            LET_AND_RETURN,
            retexpr.hir_id,
            retexpr.span,
            "returning the result of a `let` binding from a block",
            |err| {
                err.span_label(local.span, "unnecessary `let` binding");

                if let Some(src) = initexpr.span.get_source_text(cx) {
                    let sugg = if binary_expr_needs_parentheses(initexpr) {
                        if has_enclosing_paren(&src) {
                            src.to_owned()
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
                        src.to_owned()
                    };
                    err.multipart_suggestion(
                        "return the expression directly",
                        vec![(local.span, String::new()), (retexpr.span, sugg)],
                        Applicability::MachineApplicable,
                    );
                } else {
                    err.span_help(initexpr.span, "this expression can be directly returned");
                }
            },
        );
    }
}
fn last_statement_borrows<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    for_each_expr(cx, expr, |e| {
        if let Some(def_id) = fn_def_id(cx, e)
            && cx
                .tcx
                .fn_sig(def_id)
                .instantiate_identity()
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
