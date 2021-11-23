use clippy_utils::diagnostics::span_lint;
use clippy_utils::ty::implements_trait;
use clippy_utils::{self, get_trait_def_id, paths};
use if_chain::if_chain;
use rustc_hir::{BinOpKind, Expr, ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of negated comparison operators on types which only implement
    /// `PartialOrd` (e.g., `f64`).
    ///
    /// ### Why is this bad?
    /// These operators make it easy to forget that the underlying types actually allow not only three
    /// potential Orderings (Less, Equal, Greater) but also a fourth one (Uncomparable). This is
    /// especially easy to miss if the operator based comparison result is negated.
    ///
    /// ### Example
    /// ```rust
    /// use std::cmp::Ordering;
    ///
    /// // Bad
    /// let a = 1.0;
    /// let b = f64::NAN;
    ///
    /// let _not_less_or_equal = !(a <= b);
    ///
    /// // Good
    /// let a = 1.0;
    /// let b = f64::NAN;
    ///
    /// let _not_less_or_equal = match a.partial_cmp(&b) {
    ///     None | Some(Ordering::Greater) => true,
    ///     _ => false,
    /// };
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NEG_CMP_OP_ON_PARTIAL_ORD,
    complexity,
    "The use of negated comparison operators on partially ordered types may produce confusing code."
}

declare_lint_pass!(NoNegCompOpForPartialOrd => [NEG_CMP_OP_ON_PARTIAL_ORD]);

impl<'tcx> LateLintPass<'tcx> for NoNegCompOpForPartialOrd {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {

            if !in_external_macro(cx.sess(), expr.span);
            if let ExprKind::Unary(UnOp::Not, inner) = expr.kind;
            if let ExprKind::Binary(ref op, left, _) = inner.kind;
            if let BinOpKind::Le | BinOpKind::Ge | BinOpKind::Lt | BinOpKind::Gt = op.node;

            then {

                let ty = cx.typeck_results().expr_ty(left);

                let implements_ord = {
                    if let Some(id) = get_trait_def_id(cx, &paths::ORD) {
                        implements_trait(cx, ty, id, &[])
                    } else {
                        return;
                    }
                };

                let implements_partial_ord = {
                    if let Some(id) = cx.tcx.lang_items().partial_ord_trait() {
                        implements_trait(cx, ty, id, &[])
                    } else {
                        return;
                    }
                };

                if implements_partial_ord && !implements_ord {
                    span_lint(
                        cx,
                        NEG_CMP_OP_ON_PARTIAL_ORD,
                        expr.span,
                        "the use of negated comparison operators on partially ordered \
                        types produces code that is hard to read and refactor, please \
                        consider using the `partial_cmp` method instead, to make it \
                        clear that the two values could be incomparable"
                    );
                }
            }
        }
    }
}
