use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg::Sugg;
use clippy_utils::{get_parent_expr, has_ambiguous_literal_in_expr, sym};
use rustc_ast::AssignOpKind;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, PathSegment};
use rustc_lint::LateContext;
use rustc_span::source_map::Spanned;

use super::SUBOPTIMAL_FLOPS;

fn is_float_mul_expr<'a>(cx: &LateContext<'_>, expr: &'a Expr<'a>) -> Option<(&'a Expr<'a>, &'a Expr<'a>)> {
    if let ExprKind::Binary(
        Spanned {
            node: BinOpKind::Mul, ..
        },
        lhs,
        rhs,
    ) = expr.kind
        && cx.typeck_results().expr_ty(lhs).is_floating_point()
        && cx.typeck_results().expr_ty(rhs).is_floating_point()
    {
        return Some((lhs, rhs));
    }

    None
}

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>) {
    let (is_assign, op, lhs, rhs) = match &expr.kind {
        ExprKind::AssignOp(
            Spanned {
                node: AssignOpKind::AddAssign,
                ..
            },
            lhs,
            rhs,
        ) => (true, BinOpKind::Add, lhs, rhs),
        ExprKind::AssignOp(
            Spanned {
                node: AssignOpKind::SubAssign,
                ..
            },
            lhs,
            rhs,
        ) => (true, BinOpKind::Sub, lhs, rhs),
        ExprKind::Binary(
            Spanned {
                node: op @ (BinOpKind::Add | BinOpKind::Sub),
                ..
            },
            lhs,
            rhs,
        ) => (false, *op, lhs, rhs),
        _ => return,
    };

    if !is_assign
        && let Some(parent) = get_parent_expr(cx, expr)
        && let ExprKind::MethodCall(PathSegment { ident: method, .. }, receiver, ..) = parent.kind
        && method.name == sym::sqrt
        // we don't care about the applicability as this is an early-return condition
        && super::hypot::detect(cx, receiver, &mut Applicability::Unspecified).is_some()
    {
        return;
    }

    // Check if any variable in the expression has an ambiguous type (could be f32 or f64)
    // see: https://github.com/rust-lang/rust-clippy/issues/14897
    let has_ambiguous_type = |expr: &Expr<'_>| {
        (matches!(expr.kind, ExprKind::Path(_)) || matches!(expr.kind, ExprKind::Call(_, _)))
            && has_ambiguous_literal_in_expr(cx, expr)
    };

    let (recv, arg1, arg2, is_from_rhs) = if let Some((inner_lhs, inner_rhs)) = is_float_mul_expr(cx, rhs)
        && cx.typeck_results().expr_ty(lhs).is_floating_point()
        && !has_ambiguous_type(inner_lhs)
    {
        (inner_lhs, inner_rhs, lhs, true)
    } else if !is_assign
        && let Some((inner_lhs, inner_rhs)) = is_float_mul_expr(cx, lhs)
        && cx.typeck_results().expr_ty(rhs).is_floating_point()
        && !has_ambiguous_type(inner_lhs)
    {
        (inner_lhs, inner_rhs, rhs, false)
    } else {
        return;
    };

    span_lint_and_then(
        cx,
        SUBOPTIMAL_FLOPS,
        expr.span,
        "multiply and add expressions can be calculated more efficiently and accurately",
        |diag| {
            let maybe_neg_sugg = |expr, app: &mut _| {
                let sugg = Sugg::hir_with_applicability(cx, expr, "_", app);
                if let BinOpKind::Sub = op { -sugg } else { sugg }
            };
            let mut app = Applicability::MachineApplicable;
            let recv_sugg = super::lib::prepare_receiver_sugg(cx, recv, &mut app);
            let (arg1, arg2) = if is_from_rhs {
                (
                    maybe_neg_sugg(arg1, &mut app),
                    Sugg::hir_with_applicability(cx, arg2, "_", &mut app),
                )
            } else {
                (
                    Sugg::hir_with_applicability(cx, arg1, "_", &mut app),
                    maybe_neg_sugg(arg2, &mut app),
                )
            };
            diag.span_suggestion(
                expr.span,
                "consider using",
                if is_assign {
                    format!("{arg2} = {recv_sugg}.mul_add({arg1}, {arg2})")
                } else {
                    format!("{recv_sugg}.mul_add({arg1}, {arg2})")
                },
                app,
            );
        },
    );
}
