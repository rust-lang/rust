use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sugg::Sugg;
use clippy_utils::{get_parent_expr, has_ambiguous_literal_in_expr, sym};
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
    if let ExprKind::Binary(
        Spanned {
            node: op @ (BinOpKind::Add | BinOpKind::Sub),
            ..
        },
        lhs,
        rhs,
    ) = &expr.kind
    {
        if let Some(parent) = get_parent_expr(cx, expr)
            && let ExprKind::MethodCall(PathSegment { ident: method, .. }, receiver, ..) = parent.kind
            && method.name == sym::sqrt
            // we don't care about the applicability as this is an early-return condition
            && super::hypot::detect(cx, receiver, &mut Applicability::Unspecified).is_some()
        {
            return;
        }

        let maybe_neg_sugg = |expr, app: &mut _| {
            let sugg = Sugg::hir_with_applicability(cx, expr, "_", app);
            if let BinOpKind::Sub = op { -sugg } else { sugg }
        };

        let mut app = Applicability::MachineApplicable;
        let (recv, arg1, arg2) = if let Some((inner_lhs, inner_rhs)) = is_float_mul_expr(cx, lhs)
            && cx.typeck_results().expr_ty(rhs).is_floating_point()
        {
            (
                inner_lhs,
                Sugg::hir_with_applicability(cx, inner_rhs, "_", &mut app),
                maybe_neg_sugg(rhs, &mut app),
            )
        } else if let Some((inner_lhs, inner_rhs)) = is_float_mul_expr(cx, rhs)
            && cx.typeck_results().expr_ty(lhs).is_floating_point()
        {
            (
                inner_lhs,
                maybe_neg_sugg(inner_rhs, &mut app),
                Sugg::hir_with_applicability(cx, lhs, "_", &mut app),
            )
        } else {
            return;
        };

        // Check if any variable in the expression has an ambiguous type (could be f32 or f64)
        // see: https://github.com/rust-lang/rust-clippy/issues/14897
        if (matches!(recv.kind, ExprKind::Path(_)) || matches!(recv.kind, ExprKind::Call(_, _)))
            && has_ambiguous_literal_in_expr(cx, recv)
        {
            return;
        }

        span_lint_and_sugg(
            cx,
            SUBOPTIMAL_FLOPS,
            expr.span,
            "multiply and add expressions can be calculated more efficiently and accurately",
            "consider using",
            format!(
                "{}.mul_add({arg1}, {arg2})",
                super::lib::prepare_receiver_sugg(cx, recv, &mut app)
            ),
            app,
        );
    }
}
