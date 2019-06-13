
use syntax::ast::{Crate, Expr, ExprKind};
use syntax::symbol::sym;
use rustc::lint::{LintArray, LintPass, EarlyLintPass, EarlyContext};
use rustc::{declare_tool_lint, impl_lint_pass};

use if_chain::if_chain;
use crate::utils::span_help_and_lint;

declare_clippy_lint! {
    pub MAIN_RECURSION,
    pedantic,
    "function named `foo`, which is not a descriptive name"
}

pub struct MainRecursion {
    has_no_std_attr: bool
}

impl_lint_pass!(MainRecursion => [MAIN_RECURSION]);

impl MainRecursion {
    pub fn new() -> MainRecursion {
        MainRecursion {
            has_no_std_attr: false
        }
    }
}

impl EarlyLintPass for MainRecursion {
    fn check_crate(&mut self, _: &EarlyContext<'_>, krate: &Crate) {
        self.has_no_std_attr = krate.attrs.iter().any(|attr| attr.path == sym::no_std);
    }

    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if self.has_no_std_attr {
            return;
        }

        if_chain! {
            if let ExprKind::Call(func, _) = &expr.node;
            if let ExprKind::Path(_, path) = &func.node;
            if *path == sym::main;
            then {
                span_help_and_lint(
                    cx,
                    MAIN_RECURSION,
                    expr.span,
                    "You are recursing into main()",
                    "Consider using another function for this recursion"
                )
            }
        }
    }
}
