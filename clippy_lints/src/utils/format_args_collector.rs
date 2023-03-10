use clippy_utils::macros::collect_ast_format_args;
use rustc_ast::{Expr, ExprKind};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Collects [`rustc_ast::FormatArgs`] so that future late passes can call
    /// [`clippy_utils::macros::find_format_args`]
    pub FORMAT_ARGS_COLLECTOR,
    internal_warn,
    "collects `format_args` AST nodes for use in later lints"
}

declare_lint_pass!(FormatArgsCollector => [FORMAT_ARGS_COLLECTOR]);

impl EarlyLintPass for FormatArgsCollector {
    fn check_expr(&mut self, _: &EarlyContext<'_>, expr: &Expr) {
        if let ExprKind::FormatArgs(args) = &expr.kind {
            collect_ast_format_args(expr.span, args);
        }
    }
}
