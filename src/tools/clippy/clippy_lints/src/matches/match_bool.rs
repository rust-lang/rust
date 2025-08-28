use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_unit_expr;
use clippy_utils::source::expr_block;
use clippy_utils::sugg::Sugg;
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Arm, Expr, PatExprKind, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty;

use super::MATCH_BOOL;

pub(crate) fn check(cx: &LateContext<'_>, scrutinee: &Expr<'_>, arms: &[Arm<'_>], expr: &Expr<'_>) {
    // Type of expression is `bool`.
    if *cx.typeck_results().expr_ty(scrutinee).kind() == ty::Bool
        && arms
            .iter()
            .all(|arm| arm.pat.walk_short(|p| !matches!(p.kind, PatKind::Binding(..))))
    {
        span_lint_and_then(
            cx,
            MATCH_BOOL,
            expr.span,
            "`match` on a boolean expression",
            move |diag| {
                if arms.len() == 2 {
                    let mut app = Applicability::MachineApplicable;
                    let test_sugg = if let PatKind::Expr(arm_bool) = arms[0].pat.kind {
                        let test = Sugg::hir_with_applicability(cx, scrutinee, "_", &mut app);
                        if let PatExprKind::Lit { lit, .. } = arm_bool.kind {
                            match &lit.node {
                                LitKind::Bool(true) => Some(test),
                                LitKind::Bool(false) => Some(!test),
                                _ => None,
                            }
                            .map(|test| {
                                if let Some(guard) = &arms[0]
                                    .guard
                                    .map(|g| Sugg::hir_with_applicability(cx, g, "_", &mut app))
                                {
                                    test.and(guard)
                                } else {
                                    test
                                }
                            })
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let Some(test_sugg) = test_sugg {
                        let ctxt = expr.span.ctxt();
                        let (true_expr, false_expr) = (arms[0].body, arms[1].body);
                        let sugg = match (is_unit_expr(true_expr), is_unit_expr(false_expr)) {
                            (false, false) => Some(format!(
                                "if {} {} else {}",
                                test_sugg,
                                expr_block(cx, true_expr, ctxt, "..", Some(expr.span), &mut app),
                                expr_block(cx, false_expr, ctxt, "..", Some(expr.span), &mut app)
                            )),
                            (false, true) => Some(format!(
                                "if {} {}",
                                test_sugg,
                                expr_block(cx, true_expr, ctxt, "..", Some(expr.span), &mut app)
                            )),
                            (true, false) => Some(format!(
                                "if {} {}",
                                !test_sugg,
                                expr_block(cx, false_expr, ctxt, "..", Some(expr.span), &mut app)
                            )),
                            (true, true) => None,
                        };

                        if let Some(sugg) = sugg {
                            diag.span_suggestion(expr.span, "consider using an `if`/`else` expression", sugg, app);
                        }
                    }
                }
            },
        );
    }
}
