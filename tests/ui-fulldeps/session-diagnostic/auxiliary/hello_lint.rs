#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_lint;
extern crate rustc_lint_defs;
extern crate rustc_macros;
extern crate rustc_session;

use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_lint_defs::{declare_lint_pass, declare_tool_lint};
use rustc_macros::Diagnostic;

declare_tool_lint! {
    /// Test lint to test the lint testing.
    pub hello::HELLO_LINT,
    Deny,
    "warns on an binary expression"
}
declare_lint_pass!(HelloLint => [HELLO_LINT]);

#[derive(Diagnostic)]
#[diag("this is a binary expression")]
pub(crate) struct Expression;

impl LateLintPass<'_> for HelloLint {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &'_ rustc_hir::Expr<'_>) {
        if let rustc_hir::ExprKind::Binary(..) = expr.kind {
            cx.emit_span_lint(HELLO_LINT, expr.span, Expression)
        }
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn register_lints(
    _sess: &rustc_session::Session,
    lint_store: &mut rustc_lint::LintStore,
) {
    lint_store.register_lints(&[HELLO_LINT]);
    lint_store.register_late_lint_pass(Box::new(|_| Box::new(HelloLint)));
}
