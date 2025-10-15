use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{HasSession, snippet_with_applicability, snippet_with_context};
use rustc_ast::ast::{Expr, ExprKind, MethodCall};
use rustc_errors::Applicability;
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
    ///     (0)
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
        match &expr.kind {
            // ((..))
            // ^^^^^^ expr
            //  ^^^^  inner
            ExprKind::Paren(inner) if matches!(inner.kind, ExprKind::Paren(_) | ExprKind::Tup(_)) => {
                // suggest removing the outer parens
                if expr.span.eq_ctxt(inner.span) {
                    let mut applicability = Applicability::MachineApplicable;
                    // We don't need to use `snippet_with_context` here, because:
                    // - if `inner`'s `ctxt` is from macro, we don't lint in the first place (see the check above)
                    // - otherwise, calling `snippet_with_applicability` on a not-from-macro span is fine
                    let sugg = snippet_with_applicability(cx.sess(), inner.span, "_", &mut applicability);
                    span_lint_and_sugg(
                        cx,
                        DOUBLE_PARENS,
                        expr.span,
                        "unnecessary parentheses",
                        "remove them",
                        sugg.to_string(),
                        applicability,
                    );
                }
            },

            // func((n))
            // ^^^^^^^^^ expr
            //      ^^^  arg
            //       ^   inner
            ExprKind::Call(_, args) | ExprKind::MethodCall(box MethodCall { args, .. })
                if let [arg] = &**args
                    && let ExprKind::Paren(inner) = &arg.kind =>
            {
                // suggest removing the inner parens
                if expr.span.eq_ctxt(arg.span) {
                    let mut applicability = Applicability::MachineApplicable;
                    let sugg = snippet_with_context(cx.sess(), inner.span, arg.span.ctxt(), "_", &mut applicability).0;
                    span_lint_and_sugg(
                        cx,
                        DOUBLE_PARENS,
                        arg.span,
                        "unnecessary parentheses",
                        "remove them",
                        sugg.to_string(),
                        applicability,
                    );
                }
            },
            _ => {},
        }
    }
}
