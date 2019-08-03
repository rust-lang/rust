use crate::utils::{span_lint, SpanlessEq};
use if_chain::if_chain;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Detects classic underflow/overflow checks.
    ///
    /// **Why is this bad?** Most classic C underflow/overflow checks will fail in
    /// Rust. Users can use functions like `overflowing_*` and `wrapping_*` instead.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # let a = 1;
    /// # let b = 2;
    /// a + b < a;
    /// ```
    pub OVERFLOW_CHECK_CONDITIONAL,
    complexity,
    "overflow checks inspired by C which are likely to panic"
}

declare_lint_pass!(OverflowCheckConditional => [OVERFLOW_CHECK_CONDITIONAL]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for OverflowCheckConditional {
    // a + b < a, a > a + b, a < a - b, a - b > a
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        let eq = |l, r| SpanlessEq::new(cx).eq_path_segment(l, r);
        if_chain! {
            if let ExprKind::Binary(ref op, ref first, ref second) = expr.node;
            if let ExprKind::Binary(ref op2, ref ident1, ref ident2) = first.node;
            if let ExprKind::Path(QPath::Resolved(_, ref path1)) = ident1.node;
            if let ExprKind::Path(QPath::Resolved(_, ref path2)) = ident2.node;
            if let ExprKind::Path(QPath::Resolved(_, ref path3)) = second.node;
            if eq(&path1.segments[0], &path3.segments[0]) || eq(&path2.segments[0], &path3.segments[0]);
            if cx.tables.expr_ty(ident1).is_integral();
            if cx.tables.expr_ty(ident2).is_integral();
            then {
                if let BinOpKind::Lt = op.node {
                    if let BinOpKind::Add = op2.node {
                        span_lint(cx, OVERFLOW_CHECK_CONDITIONAL, expr.span,
                            "You are trying to use classic C overflow conditions that will fail in Rust.");
                    }
                }
                if let BinOpKind::Gt = op.node {
                    if let BinOpKind::Sub = op2.node {
                        span_lint(cx, OVERFLOW_CHECK_CONDITIONAL, expr.span,
                            "You are trying to use classic C underflow conditions that will fail in Rust.");
                    }
                }
            }
        }

        if_chain! {
            if let ExprKind::Binary(ref op, ref first, ref second) = expr.node;
            if let ExprKind::Binary(ref op2, ref ident1, ref ident2) = second.node;
            if let ExprKind::Path(QPath::Resolved(_, ref path1)) = ident1.node;
            if let ExprKind::Path(QPath::Resolved(_, ref path2)) = ident2.node;
            if let ExprKind::Path(QPath::Resolved(_, ref path3)) = first.node;
            if eq(&path1.segments[0], &path3.segments[0]) || eq(&path2.segments[0], &path3.segments[0]);
            if cx.tables.expr_ty(ident1).is_integral();
            if cx.tables.expr_ty(ident2).is_integral();
            then {
                if let BinOpKind::Gt = op.node {
                    if let BinOpKind::Add = op2.node {
                        span_lint(cx, OVERFLOW_CHECK_CONDITIONAL, expr.span,
                            "You are trying to use classic C overflow conditions that will fail in Rust.");
                    }
                }
                if let BinOpKind::Lt = op.node {
                    if let BinOpKind::Sub = op2.node {
                        span_lint(cx, OVERFLOW_CHECK_CONDITIONAL, expr.span,
                            "You are trying to use classic C underflow conditions that will fail in Rust.");
                    }
                }
            }
        }
    }
}
