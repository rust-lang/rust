// force-host

#![feature(plugin_registrar)]
#![feature(box_syntax, rustc_private)]

// Load rustc as a plugin to get macros.
#[macro_use]
extern crate rustc;
extern crate rustc_plugin;

use rustc::hir;
use rustc::lint::{LateContext, LintContext, LintPass, LateLintPass, LateLintPassObject, LintArray};
use rustc_plugin::Registry;

declare_lint!(TEST_LINT, Warn, "Warn about items named 'lintme'");

declare_lint!(PLEASE_LINT, Warn, "Warn about items named 'pleaselintme'");

declare_lint_pass!(Pass => [TEST_LINT, PLEASE_LINT]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_item(&mut self, cx: &LateContext, it: &hir::Item) {
        match &*it.ident.as_str() {
            "lintme" => cx.span_lint(TEST_LINT, it.span, "item is named 'lintme'"),
            "pleaselintme" => cx.span_lint(PLEASE_LINT, it.span, "item is named 'pleaselintme'"),
            _ => {}
        }
    }
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_late_lint_pass(box Pass);
    reg.register_lint_group("lint_me", None, vec![TEST_LINT, PLEASE_LINT]);
}
