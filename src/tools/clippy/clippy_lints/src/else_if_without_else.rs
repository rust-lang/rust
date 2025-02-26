use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::ast::{Expr, ExprKind};
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of if expressions with an `else if` branch,
    /// but without a final `else` branch.
    ///
    /// ### Why restrict this?
    /// Some coding guidelines require this (e.g., MISRA-C:2004 Rule 14.10).
    ///
    /// ### Example
    /// ```no_run
    /// # fn a() {}
    /// # fn b() {}
    /// # let x: i32 = 1;
    /// if x.is_positive() {
    ///     a();
    /// } else if x.is_negative() {
    ///     b();
    /// }
    /// ```
    ///
    /// Use instead:
    ///
    /// ```no_run
    /// # fn a() {}
    /// # fn b() {}
    /// # let x: i32 = 1;
    /// if x.is_positive() {
    ///     a();
    /// } else if x.is_negative() {
    ///     b();
    /// } else {
    ///     // We don't care about zero.
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub ELSE_IF_WITHOUT_ELSE,
    restriction,
    "`if` expression with an `else if`, but without a final `else` branch"
}

declare_lint_pass!(ElseIfWithoutElse => [ELSE_IF_WITHOUT_ELSE]);

impl EarlyLintPass for ElseIfWithoutElse {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, item: &Expr) {
        if let ExprKind::If(_, _, Some(ref els)) = item.kind
            && let ExprKind::If(_, _, None) = els.kind
            && !item.span.in_external_macro(cx.sess().source_map())
        {
            #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
            span_lint_and_then(
                cx,
                ELSE_IF_WITHOUT_ELSE,
                els.span,
                "`if` expression with an `else if`, but without a final `else`",
                |diag| {
                    diag.help("add an `else` block here");
                },
            );
        }
    }
}
