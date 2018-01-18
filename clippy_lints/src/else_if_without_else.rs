//! lint on if expressions with an else if, but without a final else branch

use rustc::lint::*;
use syntax::ast::*;

use utils::{in_external_macro, span_lint_and_sugg};

/// **What it does:** Checks for usage of if expressions with an `else if` branch,
/// but without a final `else` branch.
///
/// **Why is this bad?** Some coding guidelines require this (e.g. MISRA-C:2004 Rule 14.10).
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
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
/// if x.is_positive() {
///     a();
/// } else if x.is_negative() {
///     b();
/// } else {
///     // we don't care about zero
/// }
/// ```
declare_restriction_lint! {
    pub ELSE_IF_WITHOUT_ELSE,
    "if expression with an `else if`, but without a final `else` branch"
}

#[derive(Copy, Clone)]
pub struct ElseIfWithoutElse;

impl LintPass for ElseIfWithoutElse {
    fn get_lints(&self) -> LintArray {
        lint_array!(ELSE_IF_WITHOUT_ELSE)
    }
}

impl EarlyLintPass for ElseIfWithoutElse {
    fn check_expr(&mut self, cx: &EarlyContext, mut item: &Expr) {
        if in_external_macro(cx, item.span) {
            return;
        }

        while let ExprKind::If(_, _, Some(ref els)) = item.node {
            if let ExprKind::If(_, _, None) = els.node {
                span_lint_and_sugg(
                    cx,
                    ELSE_IF_WITHOUT_ELSE,
                    els.span,
                    "if expression with an `else if`, but without a final `else`",
                    "add an `else` block here",
                    "".to_string()
                );
            }

            item = els;
        }
    }
}
