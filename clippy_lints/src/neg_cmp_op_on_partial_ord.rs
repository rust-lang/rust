use if_chain::if_chain;
use rustc::hir::*;
use rustc::lint::{in_external_macro, LateContext, LateLintPass, LintArray, LintContext, LintPass};
use rustc::{declare_tool_lint, lint_array};

use crate::utils::{self, paths, span_lint};

declare_clippy_lint! {
    /// **What it does:**
    /// Checks for the usage of negated comparison operators on types which only implement
    /// `PartialOrd` (e.g., `f64`).
    ///
    /// **Why is this bad?**
    /// These operators make it easy to forget that the underlying types actually allow not only three
    /// potential Orderings (Less, Equal, Greater) but also a fourth one (Uncomparable). This is
    /// especially easy to miss if the operator based comparison result is negated.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// use std::cmp::Ordering;
    ///
    /// // Bad
    /// let a = 1.0;
    /// let b = std::f64::NAN;
    ///
    /// let _not_less_or_equal = !(a <= b);
    ///
    /// // Good
    /// let a = 1.0;
    /// let b = std::f64::NAN;
    ///
    /// let _not_less_or_equal = match a.partial_cmp(&b) {
    ///     None | Some(Ordering::Greater) => true,
    ///     _ => false,
    /// };
    /// ```
    pub NEG_CMP_OP_ON_PARTIAL_ORD,
    complexity,
    "The use of negated comparison operators on partially ordered types may produce confusing code."
}

pub struct NoNegCompOpForPartialOrd;

impl LintPass for NoNegCompOpForPartialOrd {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEG_CMP_OP_ON_PARTIAL_ORD)
    }

    fn name(&self) -> &'static str {
        "NoNegCompOpForPartialOrd"
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NoNegCompOpForPartialOrd {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_chain! {

            if !in_external_macro(cx.sess(), expr.span);
            if let ExprKind::Unary(UnOp::UnNot, ref inner) = expr.node;
            if let ExprKind::Binary(ref op, ref left, _) = inner.node;
            if let BinOpKind::Le | BinOpKind::Ge | BinOpKind::Lt | BinOpKind::Gt = op.node;

            then {

                let ty = cx.tables.expr_ty(left);

                let implements_ord = {
                    if let Some(id) = utils::get_trait_def_id(cx, &paths::ORD) {
                        utils::implements_trait(cx, ty, id, &[])
                    } else {
                        return;
                    }
                };

                let implements_partial_ord = {
                    if let Some(id) = utils::get_trait_def_id(cx, &paths::PARTIAL_ORD) {
                        utils::implements_trait(cx, ty, id, &[])
                    } else {
                        return;
                    }
                };

                if implements_partial_ord && !implements_ord {
                    span_lint(
                        cx,
                        NEG_CMP_OP_ON_PARTIAL_ORD,
                        expr.span,
                        "The use of negated comparison operators on partially ordered \
                        types produces code that is hard to read and refactor. Please \
                        consider using the `partial_cmp` method instead, to make it \
                        clear that the two values could be incomparable."
                    )
                }
            }
        }
    }
}
