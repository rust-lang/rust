use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{snippet_opt, trim_span};
use rustc_ast::ast::{Expr, ExprKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for empty `else` branches.
    ///
    /// ### Why is this bad?
    /// An empty else branch does nothing and can be removed.
    ///
    /// ### Example
    /// ```rust
    ///# fn check() -> bool { true }
    /// if check() {
    ///     println!("Check successful!");
    /// } else {
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    ///# fn check() -> bool { true }
    /// if check() {
    ///     println!("Check successful!");
    /// }
    /// ```
    #[clippy::version = "1.71.0"]
    pub NEEDLESS_ELSE,
    style,
    "empty else branch"
}
declare_lint_pass!(NeedlessElse => [NEEDLESS_ELSE]);

impl EarlyLintPass for NeedlessElse {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let ExprKind::If(_, then_block, Some(else_clause)) = &expr.kind
            && let ExprKind::Block(block, _) = &else_clause.kind
            && !expr.span.from_expansion()
            && !else_clause.span.from_expansion()
            && block.stmts.is_empty()
            && let Some(trimmed) = expr.span.trim_start(then_block.span)
            && let span = trim_span(cx.sess().source_map(), trimmed)
            && let Some(else_snippet) = snippet_opt(cx, span)
            // Ignore else blocks that contain comments or #[cfg]s
            && !else_snippet.contains(['/', '#'])
        {
            span_lint_and_sugg(
                cx,
                NEEDLESS_ELSE,
                span,
                "this `else` branch is empty",
                "you can remove it",
                String::new(),
                Applicability::MachineApplicable,
            );
        }
    }
}
