use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;

use crate::utils::*;

declare_clippy_lint! {
    /// **What it does:** Checks for expressions of the form `a * b + c`
    /// or `c + a * b` where `a`, `b`, `c` are floats and suggests using
    /// `a.mul_add(b, c)` instead.
    ///
    /// **Why is this bad?** Calculating `a * b + c` may lead to slight
    /// numerical inaccuracies as `a * b` is rounded before being added to
    /// `c`. Depending on the target architecture, `mul_add()` may be more
    /// performant.
    ///
    /// **Known problems:** This lint can emit semantic incorrect suggestions.
    /// For example, for `a * b * c + d` the suggestion `a * b.mul_add(c, d)`
    /// is emitted, which is equivalent to `a * (b * c + d)`. (#4735)
    ///
    /// **Example:**
    ///
    /// ```rust
    /// # let a = 0_f32;
    /// # let b = 0_f32;
    /// # let c = 0_f32;
    /// let foo = (a * b) + c;
    /// ```
    ///
    /// can be written as
    ///
    /// ```rust
    /// # let a = 0_f32;
    /// # let b = 0_f32;
    /// # let c = 0_f32;
    /// let foo = a.mul_add(b, c);
    /// ```
    pub MANUAL_MUL_ADD,
    nursery,
    "Using `a.mul_add(b, c)` for floating points has higher numerical precision than `a * b + c`"
}

declare_lint_pass!(MulAddCheck => [MANUAL_MUL_ADD]);

fn is_float<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &Expr) -> bool {
    cx.tables.expr_ty(expr).is_floating_point()
}

// Checks whether expression is multiplication of two floats
fn is_float_mult_expr<'a, 'tcx, 'b>(cx: &LateContext<'a, 'tcx>, expr: &'b Expr) -> Option<(&'b Expr, &'b Expr)> {
    if let ExprKind::Binary(op, lhs, rhs) = &expr.kind {
        if let BinOpKind::Mul = op.node {
            if is_float(cx, &lhs) && is_float(cx, &rhs) {
                return Some((&lhs, &rhs));
            }
        }
    }

    None
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MulAddCheck {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let ExprKind::Binary(op, lhs, rhs) = &expr.kind {
            if let BinOpKind::Add = op.node {
                //Converts mult_lhs * mult_rhs + rhs to mult_lhs.mult_add(mult_rhs, rhs)
                if let Some((mult_lhs, mult_rhs)) = is_float_mult_expr(cx, lhs) {
                    if is_float(cx, rhs) {
                        span_lint_and_sugg(
                            cx,
                            MANUAL_MUL_ADD,
                            expr.span,
                            "consider using mul_add() for better numerical precision",
                            "try",
                            format!(
                                "{}.mul_add({}, {})",
                                snippet(cx, mult_lhs.span, "_"),
                                snippet(cx, mult_rhs.span, "_"),
                                snippet(cx, rhs.span, "_"),
                            ),
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
                //Converts lhs + mult_lhs * mult_rhs to mult_lhs.mult_add(mult_rhs, lhs)
                if let Some((mult_lhs, mult_rhs)) = is_float_mult_expr(cx, rhs) {
                    if is_float(cx, lhs) {
                        span_lint_and_sugg(
                            cx,
                            MANUAL_MUL_ADD,
                            expr.span,
                            "consider using mul_add() for better numerical precision",
                            "try",
                            format!(
                                "{}.mul_add({}, {})",
                                snippet(cx, mult_lhs.span, "_"),
                                snippet(cx, mult_rhs.span, "_"),
                                snippet(cx, lhs.span, "_"),
                            ),
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
            }
        }
    }
}
