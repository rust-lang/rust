use clippy_utils::consts::is_zero_integer_const;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::expr_type_is_certain;
use rustc_ast::BinOpKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

use super::MANUAL_IS_MULTIPLE_OF;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'_>,
    op: BinOpKind,
    lhs: &'tcx Expr<'tcx>,
    rhs: &'tcx Expr<'tcx>,
    msrv: Msrv,
) {
    if msrv.meets(cx, msrvs::UNSIGNED_IS_MULTIPLE_OF)
        && let Some(operand) = uint_compare_to_zero(cx, op, lhs, rhs)
        && let ExprKind::Binary(operand_op, operand_left, operand_right) = operand.kind
        && operand_op.node == BinOpKind::Rem
        && matches!(
            cx.typeck_results().expr_ty_adjusted(operand_left).peel_refs().kind(),
            ty::Uint(_)
        )
        && matches!(
            cx.typeck_results().expr_ty_adjusted(operand_right).peel_refs().kind(),
            ty::Uint(_)
        )
        && expr_type_is_certain(cx, operand_left)
    {
        let mut app = Applicability::MachineApplicable;
        let divisor = deref_sugg(
            Sugg::hir_with_applicability(cx, operand_right, "_", &mut app),
            cx.typeck_results().expr_ty_adjusted(operand_right),
        );
        span_lint_and_sugg(
            cx,
            MANUAL_IS_MULTIPLE_OF,
            expr.span,
            "manual implementation of `.is_multiple_of()`",
            "replace with",
            format!(
                "{}{}.is_multiple_of({divisor})",
                if op == BinOpKind::Eq { "" } else { "!" },
                Sugg::hir_with_applicability(cx, operand_left, "_", &mut app).maybe_paren()
            ),
            app,
        );
    }
}

// If we have a `x == 0`, `x != 0` or `x > 0` (or the reverted ones), return the non-zero operand
fn uint_compare_to_zero<'tcx>(
    cx: &LateContext<'tcx>,
    op: BinOpKind,
    lhs: &'tcx Expr<'tcx>,
    rhs: &'tcx Expr<'tcx>,
) -> Option<&'tcx Expr<'tcx>> {
    let operand = if matches!(lhs.kind, ExprKind::Binary(..))
        && matches!(op, BinOpKind::Eq | BinOpKind::Ne | BinOpKind::Gt)
        && is_zero_integer_const(cx, rhs)
    {
        lhs
    } else if matches!(rhs.kind, ExprKind::Binary(..))
        && matches!(op, BinOpKind::Eq | BinOpKind::Ne | BinOpKind::Lt)
        && is_zero_integer_const(cx, lhs)
    {
        rhs
    } else {
        return None;
    };

    matches!(cx.typeck_results().expr_ty_adjusted(operand).kind(), ty::Uint(_)).then_some(operand)
}

fn deref_sugg<'a>(sugg: Sugg<'a>, ty: Ty<'_>) -> Sugg<'a> {
    if let ty::Ref(_, target_ty, _) = ty.kind() {
        deref_sugg(sugg.deref(), *target_ty)
    } else {
        sugg
    }
}
