use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_unit_expr;
use clippy_utils::source::{expr_block, snippet};
use clippy_utils::sugg::Sugg;
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Arm, Expr, ExprKind, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty;

use super::MATCH_BOOL;

pub(crate) fn check(cx: &LateContext<'_>, scrutinee: &Expr<'_>, arms: &[Arm<'_>], expr: &Expr<'_>) {
    // Type of expression is `bool`.
    if *cx.typeck_results().expr_ty(scrutinee).kind() == ty::Bool {
        span_lint_and_then(
            cx,
            MATCH_BOOL,
            expr.span,
            "you seem to be trying to match on a boolean expression",
            move |diag| {
                if arms.len() == 2 {
                    // no guards
                    let exprs = if let PatKind::Lit(arm_bool) = arms[0].pat.kind {
                        if let ExprKind::Lit(ref lit) = arm_bool.kind {
                            match lit.node {
                                LitKind::Bool(true) => Some((arms[0].body, arms[1].body)),
                                LitKind::Bool(false) => Some((arms[1].body, arms[0].body)),
                                _ => None,
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let Some((true_expr, false_expr)) = exprs {
                        let mut app = Applicability::HasPlaceholders;
                        let ctxt = expr.span.ctxt();
                        let sugg = match (is_unit_expr(true_expr), is_unit_expr(false_expr)) {
                            (false, false) => Some(format!(
                                "if {} {} else {}",
                                snippet(cx, scrutinee.span, "b"),
                                expr_block(cx, true_expr, ctxt, "..", Some(expr.span), &mut app),
                                expr_block(cx, false_expr, ctxt, "..", Some(expr.span), &mut app)
                            )),
                            (false, true) => Some(format!(
                                "if {} {}",
                                snippet(cx, scrutinee.span, "b"),
                                expr_block(cx, true_expr, ctxt, "..", Some(expr.span), &mut app)
                            )),
                            (true, false) => {
                                let test = Sugg::hir(cx, scrutinee, "..");
                                Some(format!(
                                    "if {} {}",
                                    !test,
                                    expr_block(cx, false_expr, ctxt, "..", Some(expr.span), &mut app)
                                ))
                            },
                            (true, true) => None,
                        };

                        if let Some(sugg) = sugg {
                            diag.span_suggestion(
                                expr.span,
                                "consider using an `if`/`else` expression",
                                sugg,
                                Applicability::HasPlaceholders,
                            );
                        }
                    }
                }
            },
        );
    }
}
