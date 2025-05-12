#![allow(clippy::match_same_arms)]

use std::cmp::Ordering;

use clippy_utils::consts::{ConstEvalCtxt, Constant};
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_middle::ty::{Ty, TypeckResults};
use rustc_span::Span;
use rustc_span::source_map::Spanned;

use clippy_utils::SpanlessEq;
use clippy_utils::diagnostics::span_lint_and_note;
use clippy_utils::source::snippet;

use super::{IMPOSSIBLE_COMPARISONS, REDUNDANT_COMPARISONS};

// Extract a comparison between a const and non-const
// Flip yoda conditionals, turnings expressions like `42 < x` into `x > 42`
fn comparison_to_const<'tcx>(
    cx: &LateContext<'tcx>,
    typeck: &'tcx TypeckResults<'tcx>,
    expr: &'tcx Expr<'tcx>,
) -> Option<(CmpOp, &'tcx Expr<'tcx>, &'tcx Expr<'tcx>, Constant<'tcx>, Ty<'tcx>)> {
    if let ExprKind::Binary(operator, left, right) = expr.kind
        && let Ok(cmp_op) = CmpOp::try_from(operator.node)
    {
        let ecx = ConstEvalCtxt::with_env(cx.tcx, cx.typing_env(), typeck);
        match (ecx.eval(left), ecx.eval(right)) {
            (Some(_), Some(_)) => None,
            (_, Some(con)) => Some((cmp_op, left, right, con, typeck.expr_ty(right))),
            (Some(con), _) => Some((cmp_op.reverse(), right, left, con, typeck.expr_ty(left))),
            _ => None,
        }
    } else {
        None
    }
}

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    and_op: Spanned<BinOpKind>,
    left_cond: &'tcx Expr<'tcx>,
    right_cond: &'tcx Expr<'tcx>,
    span: Span,
) {
    if and_op.node == BinOpKind::And
        // Ensure that the binary operator is &&

        // Check that both operands to '&&' are themselves a binary operation
        // The `comparison_to_const` step also checks this, so this step is just an optimization
        && let ExprKind::Binary(_, _, _) = left_cond.kind
        && let ExprKind::Binary(_, _, _) = right_cond.kind

        && let typeck = cx.typeck_results()

        // Check that both operands to '&&' compare a non-literal to a literal
        && let Some((left_cmp_op, left_expr, left_const_expr, left_const, left_type)) =
            comparison_to_const(cx, typeck, left_cond)
        && let Some((right_cmp_op, right_expr, right_const_expr, right_const, right_type)) =
            comparison_to_const(cx, typeck, right_cond)

        && left_type == right_type

        // Check that the same expression is compared in both comparisons
        && SpanlessEq::new(cx).eq_expr(left_expr, right_expr)

        && !left_expr.can_have_side_effects()

        // Compare the two constant expressions
        && let Some(ordering) = Constant::partial_cmp(cx.tcx(), left_type, &left_const, &right_const)

        // Rule out the `x >= 42 && x <= 42` corner case immediately
        // Mostly to simplify the implementation, but it is also covered by `clippy::double_comparisons`
        && !matches!(
            (&left_cmp_op, &right_cmp_op, ordering),
            (CmpOp::Le | CmpOp::Ge, CmpOp::Le | CmpOp::Ge, Ordering::Equal)
        )
    {
        if left_cmp_op.direction() == right_cmp_op.direction() {
            let lhs_str = snippet(cx, left_cond.span, "<lhs>");
            let rhs_str = snippet(cx, right_cond.span, "<rhs>");
            // We already know that either side of `&&` has no effect,
            // but emit a different error message depending on which side it is
            if left_side_is_useless(left_cmp_op, ordering) {
                span_lint_and_note(
                    cx,
                    REDUNDANT_COMPARISONS,
                    span,
                    "left-hand side of `&&` operator has no effect",
                    Some(left_cond.span.until(right_cond.span)),
                    format!("`if `{rhs_str}` evaluates to true, {lhs_str}` will always evaluate to true as well"),
                );
            } else {
                span_lint_and_note(
                    cx,
                    REDUNDANT_COMPARISONS,
                    span,
                    "right-hand side of `&&` operator has no effect",
                    Some(and_op.span.to(right_cond.span)),
                    format!("`if `{lhs_str}` evaluates to true, {rhs_str}` will always evaluate to true as well"),
                );
            }
            // We could autofix this error but choose not to,
            // because code triggering this lint probably not behaving correctly in the first place
        } else if !comparison_is_possible(left_cmp_op.direction(), ordering) {
            let expr_str = snippet(cx, left_expr.span, "..");
            let lhs_str = snippet(cx, left_const_expr.span, "<lhs>");
            let rhs_str = snippet(cx, right_const_expr.span, "<rhs>");
            let note = match ordering {
                Ordering::Less => format!(
                    "since `{lhs_str}` < `{rhs_str}`, the expression evaluates to false for any value of `{expr_str}`"
                ),
                Ordering::Equal => {
                    format!("`{expr_str}` cannot simultaneously be greater than and less than `{lhs_str}`")
                },
                Ordering::Greater => format!(
                    "since `{lhs_str}` > `{rhs_str}`, the expression evaluates to false for any value of `{expr_str}`"
                ),
            };
            span_lint_and_note(
                cx,
                IMPOSSIBLE_COMPARISONS,
                span,
                "boolean expression will never evaluate to 'true'",
                None,
                note,
            );
        }
    }
}

fn left_side_is_useless(left_cmp_op: CmpOp, ordering: Ordering) -> bool {
    // Special-case for equal constants with an inclusive comparison
    if ordering == Ordering::Equal {
        match left_cmp_op {
            CmpOp::Lt | CmpOp::Gt => false,
            CmpOp::Le | CmpOp::Ge => true,
        }
    } else {
        match (left_cmp_op.direction(), ordering) {
            (CmpOpDirection::Lesser, Ordering::Less) => false,
            (CmpOpDirection::Lesser, Ordering::Equal) => false,
            (CmpOpDirection::Lesser, Ordering::Greater) => true,
            (CmpOpDirection::Greater, Ordering::Less) => true,
            (CmpOpDirection::Greater, Ordering::Equal) => false,
            (CmpOpDirection::Greater, Ordering::Greater) => false,
        }
    }
}

fn comparison_is_possible(left_cmp_direction: CmpOpDirection, ordering: Ordering) -> bool {
    match (left_cmp_direction, ordering) {
        (CmpOpDirection::Lesser, Ordering::Less | Ordering::Equal) => false,
        (CmpOpDirection::Lesser, Ordering::Greater) => true,
        (CmpOpDirection::Greater, Ordering::Greater | Ordering::Equal) => false,
        (CmpOpDirection::Greater, Ordering::Less) => true,
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum CmpOpDirection {
    Lesser,
    Greater,
}

#[derive(Clone, Copy)]
enum CmpOp {
    Lt,
    Le,
    Ge,
    Gt,
}

impl CmpOp {
    fn reverse(self) -> Self {
        match self {
            CmpOp::Lt => CmpOp::Gt,
            CmpOp::Le => CmpOp::Ge,
            CmpOp::Ge => CmpOp::Le,
            CmpOp::Gt => CmpOp::Lt,
        }
    }

    fn direction(self) -> CmpOpDirection {
        match self {
            CmpOp::Lt => CmpOpDirection::Lesser,
            CmpOp::Le => CmpOpDirection::Lesser,
            CmpOp::Ge => CmpOpDirection::Greater,
            CmpOp::Gt => CmpOpDirection::Greater,
        }
    }
}

impl TryFrom<BinOpKind> for CmpOp {
    type Error = ();

    fn try_from(bin_op: BinOpKind) -> Result<Self, Self::Error> {
        match bin_op {
            BinOpKind::Lt => Ok(CmpOp::Lt),
            BinOpKind::Le => Ok(CmpOp::Le),
            BinOpKind::Ge => Ok(CmpOp::Ge),
            BinOpKind::Gt => Ok(CmpOp::Gt),
            _ => Err(()),
        }
    }
}
