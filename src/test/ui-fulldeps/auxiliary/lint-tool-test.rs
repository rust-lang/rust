#![feature(plugin_registrar)]
#![feature(box_syntax, rustc_private)]

extern crate syntax;

// Load rustc as a plugin to get macros
#[macro_use]
extern crate rustc;
extern crate rustc_driver;

use rustc::lint::{EarlyContext, EarlyLintPass, LintArray, LintContext, LintPass, LintId};
use rustc_driver::plugin::Registry;
use syntax::ast;
declare_tool_lint!(pub clippy::TEST_LINT, Warn, "Warn about stuff");
declare_tool_lint!(
    /// Some docs
    pub clippy::TEST_GROUP,
    Warn, "Warn about other stuff"
);

declare_tool_lint!(
    /// Some docs
    pub rustc::TEST_RUSTC_TOOL_LINT,
    Deny,
    "Deny internal stuff"
);

declare_lint_pass!(Pass => [TEST_LINT, TEST_GROUP, TEST_RUSTC_TOOL_LINT]);

impl EarlyLintPass for Pass {
    fn check_item(&mut self, cx: &EarlyContext, it: &ast::Item) {
        if it.ident.name.as_str() == "lintme" {
            cx.span_lint(TEST_LINT, it.span, "item is named 'lintme'");
        }
        if it.ident.name.as_str() == "lintmetoo" {
            cx.span_lint(TEST_GROUP, it.span, "item is named 'lintmetoo'");
        }
    }
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.lint_store.register_lints(&[&TEST_RUSTC_TOOL_LINT, &TEST_LINT, &TEST_GROUP]);
    reg.lint_store.register_early_pass(|| box Pass);
    reg.lint_store.register_group(true, "clippy::group", Some("clippy_group"),
        vec![LintId::of(&TEST_LINT), LintId::of(&TEST_GROUP)]);
}
