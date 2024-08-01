use clippy_utils::diagnostics::span_lint;
use rustc_ast::ast::{Expr, ExprKind};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for unnecessary double parentheses.
    ///
    /// ### Why is this bad?
    /// This makes code harder to read and might indicate a
    /// mistake.
    ///
    /// ### Example
    /// ```no_run
    /// fn simple_double_parens() -> i32 {
    ///     ((0))
    /// }
    ///
    /// # fn foo(bar: usize) {}
    /// foo((0));
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
        let span = match &expr.kind {
            ExprKind::Paren(in_paren) if matches!(in_paren.kind, ExprKind::Paren(_) | ExprKind::Tup(_)) => expr.span,
            ExprKind::Call(_, params)
                if let [param] = &**params
                    && let ExprKind::Paren(_) = param.kind =>
            {
                param.span
            },
            ExprKind::MethodCall(call)
                if let [arg] = &*call.args
                    && let ExprKind::Paren(_) = arg.kind =>
            {
                arg.span
            },
            _ => return,
        };
        if !expr.span.from_expansion() {
            span_lint(
                cx,
                DOUBLE_PARENS,
                span,
                "consider removing unnecessary double parentheses",
            );
        }
    }
}
