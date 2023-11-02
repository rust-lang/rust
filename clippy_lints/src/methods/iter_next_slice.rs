use super::utils::derefs_to_slice;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{get_parent_expr, higher};
use rustc_ast::ast;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::symbol::sym;

use super::ITER_NEXT_SLICE;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>, caller_expr: &'tcx hir::Expr<'_>) {
    // Skip lint if the `iter().next()` expression is a for loop argument,
    // since it is already covered by `&loops::ITER_NEXT_LOOP`
    let mut parent_expr_opt = get_parent_expr(cx, expr);
    while let Some(parent_expr) = parent_expr_opt {
        if higher::ForLoop::hir(parent_expr).is_some() {
            return;
        }
        parent_expr_opt = get_parent_expr(cx, parent_expr);
    }

    if derefs_to_slice(cx, caller_expr, cx.typeck_results().expr_ty(caller_expr)).is_some() {
        // caller is a Slice
        if let hir::ExprKind::Index(caller_var, index_expr, _) = &caller_expr.kind
            && let Some(higher::Range {
                start: Some(start_expr),
                end: None,
                limits: ast::RangeLimits::HalfOpen,
            }) = higher::Range::hir(index_expr)
            && let hir::ExprKind::Lit(start_lit) = &start_expr.kind
            && let ast::LitKind::Int(start_idx, _) = start_lit.node
        {
            let mut applicability = Applicability::MachineApplicable;
            let suggest = if start_idx == 0 {
                format!(
                    "{}.first()",
                    snippet_with_applicability(cx, caller_var.span, "..", &mut applicability)
                )
            } else {
                format!(
                    "{}.get({start_idx})",
                    snippet_with_applicability(cx, caller_var.span, "..", &mut applicability)
                )
            };
            span_lint_and_sugg(
                cx,
                ITER_NEXT_SLICE,
                expr.span,
                "using `.iter().next()` on a Slice without end index",
                "try calling",
                suggest,
                applicability,
            );
        }
    } else if is_vec_or_array(cx, caller_expr) {
        // caller is a Vec or an Array
        let mut applicability = Applicability::MachineApplicable;
        span_lint_and_sugg(
            cx,
            ITER_NEXT_SLICE,
            expr.span,
            "using `.iter().next()` on an array",
            "try calling",
            format!(
                "{}.first()",
                snippet_with_applicability(cx, caller_expr.span, "..", &mut applicability)
            ),
            applicability,
        );
    }
}

fn is_vec_or_array<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) -> bool {
    is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(expr), sym::Vec)
        || matches!(&cx.typeck_results().expr_ty(expr).peel_refs().kind(), ty::Array(_, _))
}
