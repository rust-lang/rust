use clippy_utils::consts::{constant_simple, Constant};
use clippy_utils::diagnostics::span_lint;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for erasing operations, e.g., `x * 0`.
    ///
    /// ### Why is this bad?
    /// The whole expression can be replaced by zero.
    /// This is most likely not the intended outcome and should probably be
    /// corrected
    ///
    /// ### Example
    /// ```rust
    /// let x = 1;
    /// 0 / x;
    /// 0 * x;
    /// x & 0;
    /// ```
    pub ERASING_OP,
    correctness,
    "using erasing operations, e.g., `x * 0` or `y & 0`"
}

declare_lint_pass!(ErasingOp => [ERASING_OP]);

impl<'tcx> LateLintPass<'tcx> for ErasingOp {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if e.span.from_expansion() {
            return;
        }
        if let ExprKind::Binary(ref cmp, left, right) = e.kind {
            match cmp.node {
                BinOpKind::Mul | BinOpKind::BitAnd => {
                    check(cx, left, e.span);
                    check(cx, right, e.span);
                },
                BinOpKind::Div => check(cx, left, e.span),
                _ => (),
            }
        }
    }
}

fn check(cx: &LateContext<'_>, e: &Expr<'_>, span: Span) {
    if constant_simple(cx, cx.typeck_results(), e) == Some(Constant::Int(0)) {
        span_lint(
            cx,
            ERASING_OP,
            span,
            "this operation will always return zero. This is likely not the intended outcome",
        );
    }
}
