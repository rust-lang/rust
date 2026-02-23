use clippy_utils::consts::ConstEvalCtxt;
use clippy_utils::consts::Constant::{F32, F64, Int};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sugg::Sugg;
use clippy_utils::{eq_expr_value, higher, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, UnOp};
use rustc_lint::LateContext;
use rustc_span::SyntaxContext;
use rustc_span::source_map::Spanned;

use super::SUBOPTIMAL_FLOPS;

/// Returns true iff expr is an expression which tests whether or not
/// test is positive or an expression which tests whether or not test
/// is nonnegative.
/// Used for check-custom-abs function below
fn is_testing_positive(cx: &LateContext<'_>, expr: &Expr<'_>, test: &Expr<'_>) -> bool {
    if let ExprKind::Binary(Spanned { node: op, .. }, left, right) = expr.kind {
        match op {
            BinOpKind::Gt | BinOpKind::Ge => is_zero(cx, right, expr.span.ctxt()) && eq_expr_value(cx, left, test),
            BinOpKind::Lt | BinOpKind::Le => is_zero(cx, left, expr.span.ctxt()) && eq_expr_value(cx, right, test),
            _ => false,
        }
    } else {
        false
    }
}

/// See [`is_testing_positive`]
fn is_testing_negative(cx: &LateContext<'_>, expr: &Expr<'_>, test: &Expr<'_>) -> bool {
    if let ExprKind::Binary(Spanned { node: op, .. }, left, right) = expr.kind {
        match op {
            BinOpKind::Gt | BinOpKind::Ge => is_zero(cx, left, expr.span.ctxt()) && eq_expr_value(cx, right, test),
            BinOpKind::Lt | BinOpKind::Le => is_zero(cx, right, expr.span.ctxt()) && eq_expr_value(cx, left, test),
            _ => false,
        }
    } else {
        false
    }
}

/// Returns true iff expr is some zero literal
fn is_zero(cx: &LateContext<'_>, expr: &Expr<'_>, ctxt: SyntaxContext) -> bool {
    match ConstEvalCtxt::new(cx).eval_local(expr, ctxt) {
        Some(Int(i)) => i == 0,
        Some(F32(f)) => f == 0.0,
        Some(F64(f)) => f == 0.0,
        _ => false,
    }
}

/// If the two expressions are negations of each other, then it returns
/// a tuple, in which the first element is true iff expr1 is the
/// positive expressions, and the second element is the positive
/// one of the two expressions
/// If the two expressions are not negations of each other, then it
/// returns None.
fn are_negated<'a>(cx: &LateContext<'_>, expr1: &'a Expr<'a>, expr2: &'a Expr<'a>) -> Option<(bool, &'a Expr<'a>)> {
    if let ExprKind::Unary(UnOp::Neg, expr1_negated) = expr1.kind
        && eq_expr_value(cx, expr1_negated, expr2)
    {
        return Some((false, expr2));
    }
    if let ExprKind::Unary(UnOp::Neg, expr2_negated) = expr2.kind
        && eq_expr_value(cx, expr1, expr2_negated)
    {
        return Some((true, expr1));
    }
    None
}

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if let Some(higher::If {
        cond,
        then,
        r#else: Some(r#else),
    }) = higher::If::hir(expr)
        && let if_body_expr = peel_blocks(then)
        && let else_body_expr = peel_blocks(r#else)
        && let Some((if_expr_positive, body)) = are_negated(cx, if_body_expr, else_body_expr)
    {
        let sugg_positive_abs = if is_testing_positive(cx, cond, body) {
            if_expr_positive
        } else if is_testing_negative(cx, cond, body) {
            !if_expr_positive
        } else {
            return;
        };
        let mut app = Applicability::MachineApplicable;
        let body = Sugg::hir_with_applicability(cx, body, "_", &mut app).maybe_paren();
        let sugg = if sugg_positive_abs {
            ("manual implementation of `abs` method", format!("{body}.abs()"))
        } else {
            #[rustfmt::skip]
            ("manual implementation of negation of `abs` method", format!("-{body}.abs()"))
        };
        span_lint_and_sugg(cx, SUBOPTIMAL_FLOPS, expr.span, sugg.0, "try", sugg.1, app);
    }
}
