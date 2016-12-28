use syntax::ast::*;
use rustc::lint::{EarlyContext, LintArray, LintPass, EarlyLintPass};

/// **What it does:** Checks for unnecessary double parentheses.
///
/// **Why is this bad?** This makes code harder to read and might indicate a
/// mistake.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// ((0))
/// foo((0))
/// ((1, 2))
/// ```
declare_lint! {
    pub DOUBLE_PARENS, Warn,
    "Warn on unnecessary double parentheses"
}

#[derive(Copy, Clone)]
pub struct DoubleParens;

impl LintPass for DoubleParens {
    fn get_lints(&self) -> LintArray {
        lint_array!(DOUBLE_PARENS)
    }
}

impl EarlyLintPass for DoubleParens {
    fn check_expr(&mut self, cx: &EarlyContext, expr: &Expr) {
        // insert check here.
    }
}
