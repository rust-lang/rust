//! lint on if branches that could be swapped so no `!` operation is necessary
//! on the condition

use rustc::lint::{in_external_macro, EarlyContext, EarlyLintPass, LintArray, LintContext, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use syntax::ast::*;

use crate::utils::span_help_and_lint;

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `!` or `!=` in an if condition with an
    /// else branch.
    ///
    /// **Why is this bad?** Negations reduce the readability of statements.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # let v: Vec<usize> = vec![];
    /// # fn a() {}
    /// # fn b() {}
    /// if !v.is_empty() {
    ///     a()
    /// } else {
    ///     b()
    /// }
    /// ```
    ///
    /// Could be written:
    ///
    /// ```rust
    /// # let v: Vec<usize> = vec![];
    /// # fn a() {}
    /// # fn b() {}
    /// if v.is_empty() {
    ///     b()
    /// } else {
    ///     a()
    /// }
    /// ```
    pub IF_NOT_ELSE,
    pedantic,
    "`if` branches that could be swapped so no negation operation is necessary on the condition"
}

declare_lint_pass!(IfNotElse => [IF_NOT_ELSE]);

impl EarlyLintPass for IfNotElse {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, item: &Expr) {
        if in_external_macro(cx.sess(), item.span) {
            return;
        }
        if let ExprKind::If(ref cond, _, Some(ref els)) = item.node {
            if let ExprKind::Block(..) = els.node {
                match cond.node {
                    ExprKind::Unary(UnOp::Not, _) => {
                        span_help_and_lint(
                            cx,
                            IF_NOT_ELSE,
                            item.span,
                            "Unnecessary boolean `not` operation",
                            "remove the `!` and swap the blocks of the if/else",
                        );
                    },
                    ExprKind::Binary(ref kind, _, _) if kind.node == BinOpKind::Ne => {
                        span_help_and_lint(
                            cx,
                            IF_NOT_ELSE,
                            item.span,
                            "Unnecessary `!=` operation",
                            "change to `==` and swap the blocks of the if/else",
                        );
                    },
                    _ => (),
                }
            }
        }
    }
}
