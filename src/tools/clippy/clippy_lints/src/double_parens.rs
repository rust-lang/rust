use clippy_utils::diagnostics::span_lint;
use rustc_ast::ast::{Expr, ExprKind};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for unnecessary double parentheses.
    ///
    /// ### Why is this bad?
    /// This makes code harder to read and might indicate a
    /// mistake.
    ///
    /// ### Example
    /// ```rust
    /// fn simple_double_parens() -> i32 {
    ///     ((0))
    /// }
    ///
    /// # fn foo(bar: usize) {}
    /// foo((0));
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// fn simple_no_parens() -> i32 {
    ///     0
    /// }
    ///
    /// # fn foo(bar: usize) {}
    /// foo(0);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DOUBLE_PARENS,
    complexity,
    "Warn on unnecessary double parentheses"
}

declare_lint_pass!(DoubleParens => [DOUBLE_PARENS]);

impl EarlyLintPass for DoubleParens {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if expr.span.from_expansion() {
            return;
        }

        let msg: &str = "consider removing unnecessary double parentheses";

        match expr.kind {
            ExprKind::Paren(ref in_paren) => match in_paren.kind {
                ExprKind::Paren(_) | ExprKind::Tup(_) => {
                    span_lint(cx, DOUBLE_PARENS, expr.span, msg);
                },
                _ => {},
            },
            ExprKind::Call(_, ref params) => {
                if params.len() == 1 {
                    let param = &params[0];
                    if let ExprKind::Paren(_) = param.kind {
                        span_lint(cx, DOUBLE_PARENS, param.span, msg);
                    }
                }
            },
            ExprKind::MethodCall(ref call) => {
                if let [ref arg] = call.args[..] {
                    if let ExprKind::Paren(_) = arg.kind {
                        span_lint(cx, DOUBLE_PARENS, arg.span, msg);
                    }
                }
            },
            _ => {},
        }
    }
}
