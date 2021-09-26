// force-host

#![feature(rustc_private)]

extern crate rustc_ast;

// Load rustc as a plugin to get macros
extern crate rustc_driver;
#[macro_use]
extern crate rustc_lint;
#[macro_use]
extern crate rustc_session;

use rustc_driver::plugin::Registry;
use rustc_lint::{EarlyContext, EarlyLintPass, LintArray, LintContext, LintPass};
use rustc_ast as ast;
declare_lint!(TEST_LINT, Warn, "Warn about items named 'lintme'");

declare_lint_pass!(Pass => [TEST_LINT]);

impl EarlyLintPass for Pass {
    fn check_item(&mut self, cx: &EarlyContext, it: &ast::Item) {
        if it.ident.name.as_str() == "lintme" {
            cx.lint(TEST_LINT, |lint| {
                lint.build("item is named 'lintme'").set_span(it.span).emit()
            });
        }
    }
}

#[no_mangle]
fn __rustc_plugin_registrar(reg: &mut Registry) {
    reg.lint_store.register_lints(&[&TEST_LINT]);
    reg.lint_store.register_early_pass(|| Box::new(Pass));
}
