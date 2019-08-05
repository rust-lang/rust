//! Lint on if expressions with an else if, but without a final else branch.

use rustc::lint::{in_external_macro, EarlyContext, EarlyLintPass, LintArray, LintContext, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use syntax::ast::*;

use crate::utils::span_help_and_lint;

declare_clippy_lint! {
    /// **What it does:** Checks for usage of if expressions with an `else if` branch,
    /// but without a final `else` branch.
    ///
    /// **Why is this bad?** Some coding guidelines require this (e.g., MISRA-C:2004 Rule 14.10).
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
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
    /// Could be written:
    ///
    /// ```rust
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
    pub ELSE_IF_WITHOUT_ELSE,
    restriction,
    "if expression with an `else if`, but without a final `else` branch"
}

declare_lint_pass!(ElseIfWithoutElse => [ELSE_IF_WITHOUT_ELSE]);

impl EarlyLintPass for ElseIfWithoutElse {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, mut item: &Expr) {
        if in_external_macro(cx.sess(), item.span) {
            return;
        }

        while let ExprKind::If(_, _, Some(ref els)) = item.node {
            if let ExprKind::If(_, _, None) = els.node {
                span_help_and_lint(
                    cx,
                    ELSE_IF_WITHOUT_ELSE,
                    els.span,
                    "if expression with an `else if`, but without a final `else`",
                    "add an `else` block here",
                );
            }

            item = els;
        }
    }
}
