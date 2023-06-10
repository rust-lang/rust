use clippy_utils::{diagnostics::span_lint_and_sugg, is_from_proc_macro, source::snippet_with_applicability};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Node};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    ///
    /// ### Why is this bad?
    ///
    /// ### Example
    /// ```rust
    /// // example code where clippy issues a warning
    /// ```
    /// Use instead:
    /// ```rust
    /// // example code which does not raise clippy warning
    /// ```
    #[clippy::version = "1.72.0"]
    pub NEEDLESS_IF,
    complexity,
    "checks for empty if branches"
}
declare_lint_pass!(NeedlessIf => [NEEDLESS_IF]);

impl LateLintPass<'_> for NeedlessIf {
    fn check_expr<'tcx>(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        if let ExprKind::If(if_expr, block, else_expr) = &expr.kind
            && let ExprKind::Block(block, ..) = block.kind
            && block.stmts.is_empty()
            && block.expr.is_none()
            && else_expr.is_none()
            && !in_external_macro(cx.sess(), expr.span)
        {
            let mut app = Applicability::MachineApplicable;
            let snippet = snippet_with_applicability(cx, if_expr.span, "{ ... }", &mut app);

            // Ignore `else if`
            if let Some(parent_id) = cx.tcx.hir().opt_parent_id(expr.hir_id)
                && let Some(Node::Expr(Expr {
                    kind: ExprKind::If(_, _, Some(else_expr)),
                    ..
                })) = cx.tcx.hir().find(parent_id)
                && else_expr.hir_id == expr.hir_id
            {
                return;
            }

            if is_from_proc_macro(cx, expr) {
                return;
            }

            span_lint_and_sugg(
                cx,
                NEEDLESS_IF,
                expr.span,
                "this if branch is empty",
                "you can remove it",
                format!("{snippet};"),
                Applicability::MachineApplicable,
            );
        }
    }
}
