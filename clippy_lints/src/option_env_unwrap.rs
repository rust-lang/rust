use crate::utils::{is_direct_expn_of, span_lint_and_help};
use if_chain::if_chain;
use rustc_ast::ast::{Expr, ExprKind};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `option_env!(...).unwrap()` and
    /// suggests usage of the `env!` macro.
    ///
    /// **Why is this bad?** Unwrapping the result of `option_env!` will panic
    /// at run-time if the environment variable doesn't exist, whereas `env!`
    /// catches it at compile-time.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust,no_run
    /// let _ = option_env!("HOME").unwrap();
    /// ```
    ///
    /// Is better expressed as:
    ///
    /// ```rust,no_run
    /// let _ = env!("HOME");
    /// ```
    pub OPTION_ENV_UNWRAP,
    correctness,
    "using `option_env!(...).unwrap()` to get environment variable"
}

declare_lint_pass!(OptionEnvUnwrap => [OPTION_ENV_UNWRAP]);

impl EarlyLintPass for OptionEnvUnwrap {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if_chain! {
            if let ExprKind::MethodCall(path_segment, args) = &expr.kind;
            let method_name = path_segment.ident.as_str();
            if method_name == "expect" || method_name == "unwrap";
            if let ExprKind::Call(caller, _) = &args[0].kind;
            if is_direct_expn_of(caller.span, "option_env").is_some();
            then {
                span_lint_and_help(
                    cx,
                    OPTION_ENV_UNWRAP,
                    expr.span,
                    "this will panic at run-time if the environment variable doesn't exist at compile-time",
                    None,
                    "consider using the `env!` macro instead"
                );
            }
        }
    }
}
