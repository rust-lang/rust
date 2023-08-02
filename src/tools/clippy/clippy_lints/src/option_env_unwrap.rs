use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::is_direct_expn_of;
use rustc_ast::ast::{Expr, ExprKind, MethodCall};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `option_env!(...).unwrap()` and
    /// suggests usage of the `env!` macro.
    ///
    /// ### Why is this bad?
    /// Unwrapping the result of `option_env!` will panic
    /// at run-time if the environment variable doesn't exist, whereas `env!`
    /// catches it at compile-time.
    ///
    /// ### Example
    /// ```rust,no_run
    /// let _ = option_env!("HOME").unwrap();
    /// ```
    ///
    /// Is better expressed as:
    ///
    /// ```rust,no_run
    /// let _ = env!("HOME");
    /// ```
    #[clippy::version = "1.43.0"]
    pub OPTION_ENV_UNWRAP,
    correctness,
    "using `option_env!(...).unwrap()` to get environment variable"
}

declare_lint_pass!(OptionEnvUnwrap => [OPTION_ENV_UNWRAP]);

impl EarlyLintPass for OptionEnvUnwrap {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        fn lint(cx: &EarlyContext<'_>, span: Span) {
            span_lint_and_help(
                cx,
                OPTION_ENV_UNWRAP,
                span,
                "this will panic at run-time if the environment variable doesn't exist at compile-time",
                None,
                "consider using the `env!` macro instead",
            );
        }

        if let ExprKind::MethodCall(box MethodCall { seg, receiver, .. }) = &expr.kind &&
		matches!(seg.ident.name, sym::expect | sym::unwrap) {
			if let ExprKind::Call(caller, _) = &receiver.kind &&
            // If it exists, it will be ::core::option::Option::Some("<env var>").unwrap() (A method call in the HIR)
            is_direct_expn_of(caller.span, "option_env").is_some() {
				lint(cx, expr.span);
			} else if let ExprKind::Path(_, caller) = &receiver.kind && // If it doesn't exist, it will be ::core::option::Option::None::<&'static str>.unwrap() (A path in the HIR)
            is_direct_expn_of(caller.span, "option_env").is_some() {
				lint(cx, expr.span);
			}
		}
    }
}
