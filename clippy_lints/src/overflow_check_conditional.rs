use clippy_utils::diagnostics::span_lint;
use clippy_utils::SpanlessEq;
use if_chain::if_chain;
use rustc_hir::{BinOpKind, Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Detects classic underflow/overflow checks.
    ///
    /// ### Why is this bad?
    /// Most classic C underflow/overflow checks will fail in
    /// Rust. Users can use functions like `overflowing_*` and `wrapping_*` instead.
    ///
    /// ### Example
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

const OVERFLOW_MSG: &str = "you are trying to use classic C overflow conditions that will fail in Rust";
const UNDERFLOW_MSG: &str = "you are trying to use classic C underflow conditions that will fail in Rust";

impl<'tcx> LateLintPass<'tcx> for OverflowCheckConditional {
    // a + b < a, a > a + b, a < a - b, a - b > a
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let eq = |l, r| SpanlessEq::new(cx).eq_path_segment(l, r);
        if_chain! {
            if let ExprKind::Binary(ref op, first, second) = expr.kind;
            if let ExprKind::Binary(ref op2, ident1, ident2) = first.kind;
            if let ExprKind::Path(QPath::Resolved(_, path1)) = ident1.kind;
            if let ExprKind::Path(QPath::Resolved(_, path2)) = ident2.kind;
            if let ExprKind::Path(QPath::Resolved(_, path3)) = second.kind;
            if eq(&path1.segments[0], &path3.segments[0]) || eq(&path2.segments[0], &path3.segments[0]);
            if cx.typeck_results().expr_ty(ident1).is_integral();
            if cx.typeck_results().expr_ty(ident2).is_integral();
            then {
                if op.node == BinOpKind::Lt && op2.node == BinOpKind::Add {
                    span_lint(cx, OVERFLOW_CHECK_CONDITIONAL, expr.span, OVERFLOW_MSG);
                }
                if op.node == BinOpKind::Gt && op2.node == BinOpKind::Sub {
                    span_lint(cx, OVERFLOW_CHECK_CONDITIONAL, expr.span, UNDERFLOW_MSG);
                }
            }
        }

        if_chain! {
            if let ExprKind::Binary(ref op, first, second) = expr.kind;
            if let ExprKind::Binary(ref op2, ident1, ident2) = second.kind;
            if let ExprKind::Path(QPath::Resolved(_, path1)) = ident1.kind;
            if let ExprKind::Path(QPath::Resolved(_, path2)) = ident2.kind;
            if let ExprKind::Path(QPath::Resolved(_, path3)) = first.kind;
            if eq(&path1.segments[0], &path3.segments[0]) || eq(&path2.segments[0], &path3.segments[0]);
            if cx.typeck_results().expr_ty(ident1).is_integral();
            if cx.typeck_results().expr_ty(ident2).is_integral();
            then {
                if op.node == BinOpKind::Gt && op2.node == BinOpKind::Add {
                    span_lint(cx, OVERFLOW_CHECK_CONDITIONAL, expr.span, OVERFLOW_MSG);
                }
                if op.node == BinOpKind::Lt && op2.node == BinOpKind::Sub {
                    span_lint(cx, OVERFLOW_CHECK_CONDITIONAL, expr.span, UNDERFLOW_MSG);
                }
            }
        }
    }
}
