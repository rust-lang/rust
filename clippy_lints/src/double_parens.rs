use crate::utils::{in_macro_or_desugar, span_lint};
use rustc::lint::{EarlyContext, EarlyLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use syntax::ast::*;

declare_clippy_lint! {
    /// **What it does:** Checks for unnecessary double parentheses.
    ///
    /// **Why is this bad?** This makes code harder to read and might indicate a
    /// mistake.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # fn foo(bar: usize) {}
    /// ((0));
    /// foo((0));
    /// ((1, 2));
    /// ```
    pub DOUBLE_PARENS,
    complexity,
    "Warn on unnecessary double parentheses"
}

declare_lint_pass!(DoubleParens => [DOUBLE_PARENS]);

impl EarlyLintPass for DoubleParens {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if in_macro_or_desugar(expr.span) {
            return;
        }

        match expr.node {
            ExprKind::Paren(ref in_paren) => match in_paren.node {
                ExprKind::Paren(_) | ExprKind::Tup(_) => {
                    span_lint(
                        cx,
                        DOUBLE_PARENS,
                        expr.span,
                        "Consider removing unnecessary double parentheses",
                    );
                },
                _ => {},
            },
            ExprKind::Call(_, ref params) => {
                if params.len() == 1 {
                    let param = &params[0];
                    if let ExprKind::Paren(_) = param.node {
                        span_lint(
                            cx,
                            DOUBLE_PARENS,
                            param.span,
                            "Consider removing unnecessary double parentheses",
                        );
                    }
                }
            },
            ExprKind::MethodCall(_, ref params) => {
                if params.len() == 2 {
                    let param = &params[1];
                    if let ExprKind::Paren(_) = param.node {
                        span_lint(
                            cx,
                            DOUBLE_PARENS,
                            param.span,
                            "Consider removing unnecessary double parentheses",
                        );
                    }
                }
            },
            _ => {},
        }
    }
}
