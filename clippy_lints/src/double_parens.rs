use syntax::ast::*;
use rustc::lint::{EarlyContext, EarlyLintPass, LintArray, LintContext, LintPass};

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
        match expr.node {
            ExprKind::Paren(ref in_paren) => match in_paren.node {
                ExprKind::Paren(_) | ExprKind::Tup(_) => {
                    cx.span_lint(DOUBLE_PARENS, expr.span, "Consider removing unnecessary double parentheses");
                },
                _ => {},
            },
            ExprKind::Call(_, ref params) => if params.len() == 1 {
                let param = &params[0];
                if let ExprKind::Paren(_) = param.node {
                    cx.span_lint(DOUBLE_PARENS, param.span, "Consider removing unnecessary double parentheses");
                }
            },
            ExprKind::MethodCall(_, ref params) => if params.len() == 2 {
                let param = &params[1];
                if let ExprKind::Paren(_) = param.node {
                    cx.span_lint(DOUBLE_PARENS, param.span, "Consider removing unnecessary double parentheses");
                }
            },
            _ => {},
        }
    }
}
