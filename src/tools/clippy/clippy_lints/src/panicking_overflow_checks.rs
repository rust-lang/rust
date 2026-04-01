use clippy_utils::diagnostics::span_lint;
use clippy_utils::eq_expr_value;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Detects C-style underflow/overflow checks.
    ///
    /// ### Why is this bad?
    /// These checks will, by default, panic in debug builds rather than check
    /// whether the operation caused an overflow.
    ///
    /// ### Example
    /// ```no_run
    /// # let a = 1i32;
    /// # let b = 2i32;
    /// if a + b < a {
    ///     // handle overflow
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let a = 1i32;
    /// # let b = 2i32;
    /// if a.checked_add(b).is_none() {
    ///     // handle overflow
    /// }
    /// ```
    ///
    /// Or:
    /// ```no_run
    /// # let a = 1i32;
    /// # let b = 2i32;
    /// if a.overflowing_add(b).1 {
    ///     // handle overflow
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub PANICKING_OVERFLOW_CHECKS,
    correctness,
    "overflow checks which will panic in debug mode"
}

declare_lint_pass!(PanickingOverflowChecks => [PANICKING_OVERFLOW_CHECKS]);

impl<'tcx> LateLintPass<'tcx> for PanickingOverflowChecks {
    // a + b < a, a > a + b, a < a - b, a - b > a
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Binary(op, lhs, rhs) = expr.kind
            && let (lt, gt) = match op.node {
                BinOpKind::Lt => (lhs, rhs),
                BinOpKind::Gt => (rhs, lhs),
                _ => return,
            }
            && let ctxt = expr.span.ctxt()
            && let (op_lhs, op_rhs, other, commutative) = match (&lt.kind, &gt.kind) {
                (&ExprKind::Binary(op, lhs, rhs), _) if op.node == BinOpKind::Add && ctxt == lt.span.ctxt() => {
                    (lhs, rhs, gt, true)
                },
                (_, &ExprKind::Binary(op, lhs, rhs)) if op.node == BinOpKind::Sub && ctxt == gt.span.ctxt() => {
                    (lhs, rhs, lt, false)
                },
                _ => return,
            }
            && let typeck = cx.typeck_results()
            && let ty = typeck.expr_ty(op_lhs)
            && matches!(ty.kind(), ty::Uint(_))
            && ty == typeck.expr_ty(op_rhs)
            && ty == typeck.expr_ty(other)
            && !expr.span.in_external_macro(cx.tcx.sess.source_map())
            && (eq_expr_value(cx, op_lhs, other) || (commutative && eq_expr_value(cx, op_rhs, other)))
        {
            span_lint(
                cx,
                PANICKING_OVERFLOW_CHECKS,
                expr.span,
                "you are trying to use classic C overflow conditions that will fail in Rust",
            );
        }
    }
}
