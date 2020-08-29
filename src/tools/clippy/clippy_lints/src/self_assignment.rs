use crate::utils::{eq_expr_value, snippet, span_lint};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for explicit self-assignments.
    ///
    /// **Why is this bad?** Self-assignments are redundant and unlikely to be
    /// intentional.
    ///
    /// **Known problems:** If expression contains any deref coercions or
    /// indexing operations they are assumed not to have any side effects.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// struct Event {
    ///     id: usize,
    ///     x: i32,
    ///     y: i32,
    /// }
    ///
    /// fn copy_position(a: &mut Event, b: &Event) {
    ///     a.x = b.x;
    ///     a.y = a.y;
    /// }
    /// ```
    pub SELF_ASSIGNMENT,
    correctness,
    "explicit self-assignment"
}

declare_lint_pass!(SelfAssignment => [SELF_ASSIGNMENT]);

impl<'tcx> LateLintPass<'tcx> for SelfAssignment {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Assign(lhs, rhs, _) = &expr.kind {
            if eq_expr_value(cx, lhs, rhs) {
                let lhs = snippet(cx, lhs.span, "<lhs>");
                let rhs = snippet(cx, rhs.span, "<rhs>");
                span_lint(
                    cx,
                    SELF_ASSIGNMENT,
                    expr.span,
                    &format!("self-assignment of `{}` to `{}`", rhs, lhs),
                );
            }
        }
    }
}
