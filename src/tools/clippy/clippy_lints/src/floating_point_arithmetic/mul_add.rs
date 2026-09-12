use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::expr_type_is_certain;
use clippy_utils::{get_parent_expr, sym};
use rustc_ast::AssignOpKind;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, PathSegment};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::Spanned;

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

    let (recv, arg1, arg2, is_from_rhs, lhs_typ) = if let Some((inner_lhs, inner_rhs)) = is_float_mul_expr(cx, rhs)
        && let ty::Float(float_ty) = cx.typeck_results().expr_ty(lhs).kind()
    {
        (inner_lhs, inner_rhs, lhs, true, float_ty)
    } else if !is_assign
        && let Some((inner_lhs, inner_rhs)) = is_float_mul_expr(cx, lhs)
        && let ty::Float(float_ty) = cx.typeck_results().expr_ty(rhs).kind()
    {
        (inner_lhs, inner_rhs, rhs, false, float_ty)
    } else {
        return;
    };

    span_lint_and_then(
        cx,
        SUBOPTIMAL_FLOPS,
        expr.span,
        "multiply and add expressions may be calculated more efficiently and accurately",
        |diag| {
            let maybe_neg_sugg = |expr, app: &mut _| {
                let sugg = Sugg::hir_with_applicability(cx, expr, "_", app);
                if let BinOpKind::Sub = op { -sugg } else { sugg }
            };
            let mut app = Applicability::MachineApplicable;

            let (recv_sugg, suffix_was_added) = super::lib::prepare_receiver_sugg(cx, recv, &mut app);

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

            let mul_add_call = if suffix_was_added || expr_type_is_certain(cx, recv) {
                format!("{recv_sugg}.mul_add({arg1}, {arg2})")
            } else {
                // If the receiver contains an ambiguous literal, we need to call `mul_add` with its inferred type.
                format!("{}::mul_add({recv_sugg}, {arg1}, {arg2})", lhs_typ.name_str())
            };

            diag.span_suggestion(
                expr.span,
                "consider using",
                if is_assign {
                    format!("{arg2} = {mul_add_call}")
                } else {
                    mul_add_call
                },
                app,
            );
            diag.note_once("the performance gain from `mul_add` may vary depending on the target architecture");
        },
    );
}
