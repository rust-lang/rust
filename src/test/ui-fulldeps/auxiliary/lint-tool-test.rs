#![feature(plugin_registrar)]
#![feature(box_syntax, rustc_private)]

extern crate syntax;

// Load rustc as a plugin to get macros
#[macro_use]
extern crate rustc;
extern crate rustc_plugin;

use rustc::lint::{EarlyContext, LintContext, LintPass, EarlyLintPass,
                  LintArray};
use rustc_plugin::Registry;
use syntax::ast;
declare_tool_lint!(pub clippy::TEST_LINT, Warn, "Warn about stuff");
declare_tool_lint!(
    /// Some docs
    pub clippy::TEST_GROUP,
    Warn, "Warn about other stuff"
);

declare_lint_pass!(Pass => [TEST_LINT, TEST_GROUP]);

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
    reg.register_early_lint_pass(box Pass);
    reg.register_lint_group("clippy::group", Some("clippy_group"), vec![TEST_LINT, TEST_GROUP]);
}
