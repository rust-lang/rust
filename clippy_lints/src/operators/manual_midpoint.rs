use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::sugg::Sugg;
use clippy_utils::{is_float_literal, is_integer_literal};
use rustc_ast::BinOpKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

use super::MANUAL_MIDPOINT;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    op: BinOpKind,
    left: &'tcx Expr<'_>,
    right: &'tcx Expr<'_>,
    msrv: Msrv,
) {
    let (maybe_add_expr, require_uint) =
        if op == BinOpKind::Div && (is_integer_literal(right, 2) || is_float_literal(right, 2.0)) {
            (left, false)
        } else if op == BinOpKind::Mul && is_float_literal(left, 0.5) {
            (right, false)
        } else if op == BinOpKind::Mul && is_float_literal(right, 0.5) {
            (left, false)
        } else if op == BinOpKind::Shr && is_integer_literal(right, 1) {
            (left, true)
        } else {
            return;
        };

    if !left.span.from_expansion()
        && !right.span.from_expansion()
        && let Some((add_l_expr, add_r_expr)) = add_operands(maybe_add_expr)
        && add_operands(add_l_expr).is_none() && add_operands(add_r_expr).is_none()
        && let left_ty = cx.typeck_results().expr_ty_adjusted(add_l_expr)
        && let right_ty = cx.typeck_results().expr_ty_adjusted(add_r_expr)
        && left_ty == right_ty
        && (!require_uint || matches!(left_ty.kind(), ty::Uint(_)))
        // Do not lint on `(_+1)/2` and `(1+_)/2`, it is likely a `div_ceil()` operation
        && !is_integer_literal(add_l_expr, 1) && !is_integer_literal(add_r_expr, 1)
        && is_midpoint_implemented(cx, left_ty, msrv)
    {
        let mut app = Applicability::MachineApplicable;
        let left_sugg = Sugg::hir_with_context(cx, add_l_expr, expr.span.ctxt(), "..", &mut app);
        let right_sugg = Sugg::hir_with_context(cx, add_r_expr, expr.span.ctxt(), "..", &mut app);
        let sugg = format!("{left_ty}::midpoint({left_sugg}, {right_sugg})");
        span_lint_and_sugg(
            cx,
            MANUAL_MIDPOINT,
            expr.span,
            "manual implementation of `midpoint` which can overflow",
            format!("use `{left_ty}::midpoint` instead"),
            sugg,
            app,
        );
    }
}

/// Return the left and right operands if `expr` represents an addition
fn add_operands<'e, 'tcx>(expr: &'e Expr<'tcx>) -> Option<(&'e Expr<'tcx>, &'e Expr<'tcx>)> {
    match expr.kind {
        ExprKind::Binary(op, left, right) if op.node == BinOpKind::Add => Some((left, right)),
        _ => None,
    }
}

fn is_midpoint_implemented(cx: &LateContext<'_>, ty: Ty<'_>, msrv: Msrv) -> bool {
    match ty.kind() {
        ty::Uint(_) | ty::Float(_) => msrv.meets(cx, msrvs::UINT_FLOAT_MIDPOINT),
        ty::Int(_) => msrv.meets(cx, msrvs::INT_MIDPOINT),
        _ => false,
    }
}
