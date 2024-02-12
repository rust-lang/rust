use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sext;
use rustc_hir::{BinOpKind, Expr, ExprKind, Node};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use std::fmt::Display;

use super::MODULO_ARITHMETIC;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    op: BinOpKind,
    lhs: &'tcx Expr<'_>,
    rhs: &'tcx Expr<'_>,
    allow_comparison_to_zero: bool,
) {
    if op == BinOpKind::Rem {
        if allow_comparison_to_zero && used_in_comparison_with_zero(cx, e) {
            return;
        }

        let lhs_operand = analyze_operand(lhs, cx, e);
        let rhs_operand = analyze_operand(rhs, cx, e);
        if let Some(lhs_operand) = lhs_operand
            && let Some(rhs_operand) = rhs_operand
        {
            check_const_operands(cx, e, &lhs_operand, &rhs_operand);
        } else {
            check_non_const_operands(cx, e, lhs);
        }
    };
}

fn used_in_comparison_with_zero(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let Node::Expr(parent_expr) = cx.tcx.parent_hir_node(expr.hir_id) else {
        return false;
    };
    let ExprKind::Binary(op, lhs, rhs) = parent_expr.kind else {
        return false;
    };

    if op.node == BinOpKind::Eq || op.node == BinOpKind::Ne {
        if let Some(Constant::Int(0)) = constant(cx, cx.typeck_results(), rhs) {
            return true;
        }
        if let Some(Constant::Int(0)) = constant(cx, cx.typeck_results(), lhs) {
            return true;
        }
    }

    false
}

struct OperandInfo {
    string_representation: Option<String>,
    is_negative: bool,
    is_integral: bool,
}

fn analyze_operand(operand: &Expr<'_>, cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<OperandInfo> {
    match constant(cx, cx.typeck_results(), operand) {
        Some(Constant::Int(v)) => match *cx.typeck_results().expr_ty(expr).kind() {
            ty::Int(ity) => {
                let value = sext(cx.tcx, v, ity);
                return Some(OperandInfo {
                    string_representation: Some(value.to_string()),
                    is_negative: value < 0,
                    is_integral: true,
                });
            },
            ty::Uint(_) => {
                return Some(OperandInfo {
                    string_representation: None,
                    is_negative: false,
                    is_integral: true,
                });
            },
            _ => {},
        },
        Some(Constant::F32(f)) => {
            return Some(floating_point_operand_info(&f));
        },
        Some(Constant::F64(f)) => {
            return Some(floating_point_operand_info(&f));
        },
        _ => {},
    }
    None
}

fn floating_point_operand_info<T: Display + PartialOrd + From<f32>>(f: &T) -> OperandInfo {
    OperandInfo {
        string_representation: Some(format!("{:.3}", *f)),
        is_negative: *f < 0.0.into(),
        is_integral: false,
    }
}

fn might_have_negative_value(t: Ty<'_>) -> bool {
    t.is_signed() || t.is_floating_point()
}

fn check_const_operands<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    lhs_operand: &OperandInfo,
    rhs_operand: &OperandInfo,
) {
    if lhs_operand.is_negative ^ rhs_operand.is_negative {
        span_lint_and_then(
            cx,
            MODULO_ARITHMETIC,
            expr.span,
            &format!(
                "you are using modulo operator on constants with different signs: `{} % {}`",
                lhs_operand.string_representation.as_ref().unwrap(),
                rhs_operand.string_representation.as_ref().unwrap()
            ),
            |diag| {
                diag.note("double check for expected result especially when interoperating with different languages");
                if lhs_operand.is_integral {
                    diag.note("or consider using `rem_euclid` or similar function");
                }
            },
        );
    }
}

fn check_non_const_operands<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, operand: &Expr<'_>) {
    let operand_type = cx.typeck_results().expr_ty(operand);
    if might_have_negative_value(operand_type) {
        span_lint_and_then(
            cx,
            MODULO_ARITHMETIC,
            expr.span,
            "you are using modulo operator on types that might have different signs",
            |diag| {
                diag.note("double check for expected result especially when interoperating with different languages");
                if operand_type.is_integral() {
                    diag.note("or consider using `rem_euclid` or similar function");
                }
            },
        );
    }
}
