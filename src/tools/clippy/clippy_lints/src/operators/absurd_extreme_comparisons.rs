use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty;

use clippy_utils::comparisons::{Rel, normalize_comparison};
use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::source::snippet;
use clippy_utils::ty::is_isize_or_usize;
use clippy_utils::{clip, int_bits, unsext};

use super::ABSURD_EXTREME_COMPARISONS;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    op: BinOpKind,
    lhs: &'tcx Expr<'_>,
    rhs: &'tcx Expr<'_>,
) {
    if let Some((culprit, result)) = detect_absurd_comparison(cx, op, lhs, rhs) {
        let msg = "this comparison involving the minimum or maximum element for this \
                           type contains a case that is always true or always false";

        let conclusion = match result {
            AbsurdComparisonResult::AlwaysFalse => "this comparison is always false".to_owned(),
            AbsurdComparisonResult::AlwaysTrue => "this comparison is always true".to_owned(),
            AbsurdComparisonResult::InequalityImpossible => format!(
                "the case where the two sides are not equal never occurs, consider using `{} == {}` \
                         instead",
                snippet(cx, lhs.span, "lhs"),
                snippet(cx, rhs.span, "rhs")
            ),
        };

        let help = format!(
            "because `{}` is the {} value for this type, {conclusion}",
            snippet(cx, culprit.expr.span, "x"),
            match culprit.which {
                ExtremeType::Minimum => "minimum",
                ExtremeType::Maximum => "maximum",
            }
        );

        span_lint_and_help(cx, ABSURD_EXTREME_COMPARISONS, expr.span, msg, None, help);
    }
}

enum ExtremeType {
    Minimum,
    Maximum,
}

struct ExtremeExpr<'a> {
    which: ExtremeType,
    expr: &'a Expr<'a>,
}

enum AbsurdComparisonResult {
    AlwaysFalse,
    AlwaysTrue,
    InequalityImpossible,
}

fn is_cast_between_fixed_and_target<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    if let ExprKind::Cast(cast_exp, _) = expr.kind {
        let precast_ty = cx.typeck_results().expr_ty(cast_exp);
        let cast_ty = cx.typeck_results().expr_ty(expr);

        return is_isize_or_usize(precast_ty) != is_isize_or_usize(cast_ty);
    }

    false
}

fn detect_absurd_comparison<'tcx>(
    cx: &LateContext<'tcx>,
    op: BinOpKind,
    lhs: &'tcx Expr<'_>,
    rhs: &'tcx Expr<'_>,
) -> Option<(ExtremeExpr<'tcx>, AbsurdComparisonResult)> {
    use AbsurdComparisonResult::{AlwaysFalse, AlwaysTrue, InequalityImpossible};
    use ExtremeType::{Maximum, Minimum};
    // absurd comparison only makes sense on primitive types
    // primitive types don't implement comparison operators with each other
    if cx.typeck_results().expr_ty(lhs) != cx.typeck_results().expr_ty(rhs) {
        return None;
    }

    // comparisons between fix sized types and target sized types are considered unanalyzable
    if is_cast_between_fixed_and_target(cx, lhs) || is_cast_between_fixed_and_target(cx, rhs) {
        return None;
    }

    let (rel, normalized_lhs, normalized_rhs) = normalize_comparison(op, lhs, rhs)?;

    let lx = detect_extreme_expr(cx, normalized_lhs);
    let rx = detect_extreme_expr(cx, normalized_rhs);

    Some(match rel {
        Rel::Lt => {
            match (lx, rx) {
                (Some(l @ ExtremeExpr { which: Maximum, .. }), _) => (l, AlwaysFalse), // max < x
                (_, Some(r @ ExtremeExpr { which: Minimum, .. })) => (r, AlwaysFalse), // x < min
                _ => return None,
            }
        },
        Rel::Le => {
            match (lx, rx) {
                (Some(l @ ExtremeExpr { which: Minimum, .. }), _) => (l, AlwaysTrue), // min <= x
                (Some(l @ ExtremeExpr { which: Maximum, .. }), _) => (l, InequalityImpossible), // max <= x
                (_, Some(r @ ExtremeExpr { which: Minimum, .. })) => (r, InequalityImpossible), // x <= min
                (_, Some(r @ ExtremeExpr { which: Maximum, .. })) => (r, AlwaysTrue), // x <= max
                _ => return None,
            }
        },
        Rel::Ne | Rel::Eq => return None,
    })
}

fn detect_extreme_expr<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<ExtremeExpr<'tcx>> {
    let ty = cx.typeck_results().expr_ty(expr);

    let cv = ConstEvalCtxt::new(cx).eval(expr)?;

    let which = match (ty.kind(), cv) {
        (&ty::Bool, Constant::Bool(false)) | (&ty::Uint(_), Constant::Int(0)) => ExtremeType::Minimum,
        (&ty::Int(ity), Constant::Int(i)) if i == unsext(cx.tcx, i128::MIN >> (128 - int_bits(cx.tcx, ity)), ity) => {
            ExtremeType::Minimum
        },

        (&ty::Bool, Constant::Bool(true)) => ExtremeType::Maximum,
        (&ty::Int(ity), Constant::Int(i)) if i == unsext(cx.tcx, i128::MAX >> (128 - int_bits(cx.tcx, ity)), ity) => {
            ExtremeType::Maximum
        },
        (&ty::Uint(uty), Constant::Int(i)) if clip(cx.tcx, u128::MAX, uty) == i => ExtremeType::Maximum,

        _ => return None,
    };
    Some(ExtremeExpr { which, expr })
}
