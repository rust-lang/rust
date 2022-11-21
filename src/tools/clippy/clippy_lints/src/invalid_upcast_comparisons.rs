use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, IntTy, UintTy};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

use clippy_utils::comparisons;
use clippy_utils::comparisons::Rel;
use clippy_utils::consts::{constant_full_int, FullInt};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::source::snippet;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for comparisons where the relation is always either
    /// true or false, but where one side has been upcast so that the comparison is
    /// necessary. Only integer types are checked.
    ///
    /// ### Why is this bad?
    /// An expression like `let x : u8 = ...; (x as u32) > 300`
    /// will mistakenly imply that it is possible for `x` to be outside the range of
    /// `u8`.
    ///
    /// ### Known problems
    /// https://github.com/rust-lang/rust-clippy/issues/886
    ///
    /// ### Example
    /// ```rust
    /// let x: u8 = 1;
    /// (x as u32) > 300;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub INVALID_UPCAST_COMPARISONS,
    pedantic,
    "a comparison involving an upcast which is always true or false"
}

declare_lint_pass!(InvalidUpcastComparisons => [INVALID_UPCAST_COMPARISONS]);

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

fn err_upcast_comparison(cx: &LateContext<'_>, span: Span, expr: &Expr<'_>, always: bool) {
    if let ExprKind::Cast(cast_val, _) = expr.kind {
        span_lint(
            cx,
            INVALID_UPCAST_COMPARISONS,
            span,
            &format!(
                "because of the numeric bounds on `{}` prior to casting, this expression is always {}",
                snippet(cx, cast_val.span, "the expression"),
                if always { "true" } else { "false" },
            ),
        );
    }
}

fn upcast_comparison_bounds_err<'tcx>(
    cx: &LateContext<'tcx>,
    span: Span,
    rel: comparisons::Rel,
    lhs_bounds: Option<(FullInt, FullInt)>,
    lhs: &'tcx Expr<'_>,
    rhs: &'tcx Expr<'_>,
    invert: bool,
) {
    if let Some((lb, ub)) = lhs_bounds {
        if let Some(norm_rhs_val) = constant_full_int(cx, cx.typeck_results(), rhs) {
            if rel == Rel::Eq || rel == Rel::Ne {
                if norm_rhs_val < lb || norm_rhs_val > ub {
                    err_upcast_comparison(cx, span, lhs, rel == Rel::Ne);
                }
            } else if match rel {
                Rel::Lt => {
                    if invert {
                        norm_rhs_val < lb
                    } else {
                        ub < norm_rhs_val
                    }
                },
                Rel::Le => {
                    if invert {
                        norm_rhs_val <= lb
                    } else {
                        ub <= norm_rhs_val
                    }
                },
                Rel::Eq | Rel::Ne => unreachable!(),
            } {
                err_upcast_comparison(cx, span, lhs, true);
            } else if match rel {
                Rel::Lt => {
                    if invert {
                        norm_rhs_val >= ub
                    } else {
                        lb >= norm_rhs_val
                    }
                },
                Rel::Le => {
                    if invert {
                        norm_rhs_val > ub
                    } else {
                        lb > norm_rhs_val
                    }
                },
                Rel::Eq | Rel::Ne => unreachable!(),
            } {
                err_upcast_comparison(cx, span, lhs, false);
            }
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for InvalidUpcastComparisons {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Binary(ref cmp, lhs, rhs) = expr.kind {
            let normalized = comparisons::normalize_comparison(cmp.node, lhs, rhs);
            let Some((rel, normalized_lhs, normalized_rhs)) = normalized else {
                return;
            };

            let lhs_bounds = numeric_cast_precast_bounds(cx, normalized_lhs);
            let rhs_bounds = numeric_cast_precast_bounds(cx, normalized_rhs);

            upcast_comparison_bounds_err(cx, expr.span, rel, lhs_bounds, normalized_lhs, normalized_rhs, false);
            upcast_comparison_bounds_err(cx, expr.span, rel, rhs_bounds, normalized_rhs, normalized_lhs, true);
        }
    }
}
