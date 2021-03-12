use crate::methods::derefs_to_slice;
use crate::utils::{get_parent_expr, higher, is_type_diagnostic_item, snippet_with_applicability, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_ast::ast;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::symbol::sym;

use super::ITER_NEXT_SLICE;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>, iter_args: &'tcx [hir::Expr<'_>]) {
    let caller_expr = &iter_args[0];

    // Skip lint if the `iter().next()` expression is a for loop argument,
    // since it is already covered by `&loops::ITER_NEXT_LOOP`
    let mut parent_expr_opt = get_parent_expr(cx, expr);
    while let Some(parent_expr) = parent_expr_opt {
        if higher::for_loop(parent_expr).is_some() {
            return;
        }
        parent_expr_opt = get_parent_expr(cx, parent_expr);
    }

    if derefs_to_slice(cx, caller_expr, cx.typeck_results().expr_ty(caller_expr)).is_some() {
        // caller is a Slice
        if_chain! {
            if let hir::ExprKind::Index(ref caller_var, ref index_expr) = &caller_expr.kind;
            if let Some(higher::Range { start: Some(start_expr), end: None, limits: ast::RangeLimits::HalfOpen })
                = higher::range(index_expr);
            if let hir::ExprKind::Lit(ref start_lit) = &start_expr.kind;
            if let ast::LitKind::Int(start_idx, _) = start_lit.node;
            then {
                let mut applicability = Applicability::MachineApplicable;
                span_lint_and_sugg(
                    cx,
                    ITER_NEXT_SLICE,
                    expr.span,
                    "using `.iter().next()` on a Slice without end index",
                    "try calling",
                    format!("{}.get({})", snippet_with_applicability(cx, caller_var.span, "..", &mut applicability), start_idx),
                    applicability,
                );
            }
        }
    } else if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(caller_expr), sym::vec_type)
        || matches!(
            &cx.typeck_results().expr_ty(caller_expr).peel_refs().kind(),
            ty::Array(_, _)
        )
    {
        // caller is a Vec or an Array
        let mut applicability = Applicability::MachineApplicable;
        span_lint_and_sugg(
            cx,
            ITER_NEXT_SLICE,
            expr.span,
            "using `.iter().next()` on an array",
            "try calling",
            format!(
                "{}.get(0)",
                snippet_with_applicability(cx, caller_expr.span, "..", &mut applicability)
            ),
            applicability,
        );
    }
}
