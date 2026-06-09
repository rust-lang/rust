use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, IntTy, UintTy};
use rustc_span::Span;

use clippy_utils::comparisons;
use clippy_utils::comparisons::Rel;
use clippy_utils::consts::{ConstEvalCtxt, FullInt};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::source::snippet_with_context;

use super::INVALID_UPCAST_COMPARISONS;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    cmp: BinOpKind,
    lhs: &'tcx Expr<'_>,
    rhs: &'tcx Expr<'_>,
    span: Span,
) {
    let normalized = comparisons::normalize_comparison(cmp, lhs, rhs);
    let Some((rel, normalized_lhs, normalized_rhs)) = normalized else {
        return;
    };

    let lhs_bounds = numeric_cast_precast_bounds(cx, normalized_lhs);
    let rhs_bounds = numeric_cast_precast_bounds(cx, normalized_rhs);

    upcast_comparison_bounds_err(cx, span, rel, lhs_bounds, normalized_lhs, normalized_rhs, false);
    upcast_comparison_bounds_err(cx, span, rel, rhs_bounds, normalized_rhs, normalized_lhs, true);
}

fn numeric_cast_precast_bounds(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<(FullInt, FullInt)> {
    if let ExprKind::Cast(cast_exp, _) = expr.kind {
        let pre_cast_ty = cx.typeck_results().expr_ty(cast_exp);
        let cast_ty = cx.typeck_results().expr_ty(expr);
        // if it's a cast from i32 to u32 wrapping will invalidate all these checks
        if cx.layout_of(pre_cast_ty).ok().map(|l| l.size) == cx.layout_of(cast_ty).ok().map(|l| l.size) {
            return None;
        }
        match pre_cast_ty.kind() {
            ty::Int(int_ty) => Some(match int_ty {
                IntTy::I8 => (FullInt::S(i128::from(i8::MIN)), FullInt::S(i128::from(i8::MAX))),
                IntTy::I16 => (FullInt::S(i128::from(i16::MIN)), FullInt::S(i128::from(i16::MAX))),
                IntTy::I32 => (FullInt::S(i128::from(i32::MIN)), FullInt::S(i128::from(i32::MAX))),
                IntTy::I64 => (FullInt::S(i128::from(i64::MIN)), FullInt::S(i128::from(i64::MAX))),
                IntTy::I128 => (FullInt::S(i128::MIN), FullInt::S(i128::MAX)),
                IntTy::Isize => (FullInt::S(isize::MIN as i128), FullInt::S(isize::MAX as i128)),
            }),
            ty::Uint(uint_ty) => Some(match uint_ty {
                UintTy::U8 => (FullInt::U(u128::from(u8::MIN)), FullInt::U(u128::from(u8::MAX))),
                UintTy::U16 => (FullInt::U(u128::from(u16::MIN)), FullInt::U(u128::from(u16::MAX))),
                UintTy::U32 => (FullInt::U(u128::from(u32::MIN)), FullInt::U(u128::from(u32::MAX))),
                UintTy::U64 => (FullInt::U(u128::from(u64::MIN)), FullInt::U(u128::from(u64::MAX))),
                UintTy::U128 => (FullInt::U(u128::MIN), FullInt::U(u128::MAX)),
                UintTy::Usize => (FullInt::U(usize::MIN as u128), FullInt::U(usize::MAX as u128)),
            }),
            _ => None,
        }
    } else {
        None
    }
}

fn upcast_comparison_bounds_err<'tcx>(
    cx: &LateContext<'tcx>,
    span: Span,
    rel: Rel,
    lhs_bounds: Option<(FullInt, FullInt)>,
    lhs: &'tcx Expr<'_>,
    rhs: &'tcx Expr<'_>,
    invert: bool,
) {
    if let Some((lb, ub)) = lhs_bounds
        && let Some(norm_rhs_val) = ConstEvalCtxt::new(cx).eval_full_int(rhs, span.ctxt())
    {
        match rel {
            Rel::Eq => {
                if norm_rhs_val < lb || ub < norm_rhs_val {
                    err_upcast_comparison(cx, span, lhs, false);
                }
            },
            Rel::Ne => {
                if norm_rhs_val < lb || ub < norm_rhs_val {
                    err_upcast_comparison(cx, span, lhs, true);
                }
            },
            Rel::Lt => {
                if (invert && norm_rhs_val < lb) || (!invert && ub < norm_rhs_val) {
                    err_upcast_comparison(cx, span, lhs, true);
                } else if (!invert && norm_rhs_val <= lb) || (invert && ub <= norm_rhs_val) {
                    err_upcast_comparison(cx, span, lhs, false);
                }
            },
            Rel::Le => {
                if (invert && norm_rhs_val <= lb) || (!invert && ub <= norm_rhs_val) {
                    err_upcast_comparison(cx, span, lhs, true);
                } else if (!invert && norm_rhs_val < lb) || (invert && ub < norm_rhs_val) {
                    err_upcast_comparison(cx, span, lhs, false);
                }
            },
        }
    }
}

fn err_upcast_comparison(cx: &LateContext<'_>, span: Span, expr: &Expr<'_>, always: bool) {
    if let ExprKind::Cast(cast_val, _) = expr.kind {
        let mut applicability = Applicability::MachineApplicable;
        let (cast_val_snip, _) = snippet_with_context(
            cx,
            cast_val.span,
            expr.span.ctxt(),
            "the expression",
            &mut applicability,
        );
        span_lint(
            cx,
            INVALID_UPCAST_COMPARISONS,
            span,
            format!(
                "because of the numeric bounds on `{}` prior to casting, this expression is always {}",
                cast_val_snip,
                if always { "true" } else { "false" },
            ),
        );
    }
}
