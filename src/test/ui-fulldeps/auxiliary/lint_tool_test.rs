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
declare_tool_lint!(pub clippy::TEST_GROUP, Warn, "Warn about other stuff");

struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(TEST_LINT, TEST_GROUP)
    }
}

impl EarlyLintPass for Pass {
    fn check_item(&mut self, cx: &EarlyContext, it: &ast::Item) {
        if it.ident.name == "lintme" {
            cx.span_lint(TEST_LINT, it.span, "item is named 'lintme'");
        }
        if it.ident.name == "lintmetoo" {
            cx.span_lint(TEST_GROUP, it.span, "item is named 'lintmetoo'");
        }
    }
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_early_lint_pass(box Pass);
    reg.register_lint_group("clippy::group", Some("clippy_group"), vec![TEST_LINT, TEST_GROUP]);
}
