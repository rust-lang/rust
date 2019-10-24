// force-host

#![feature(plugin_registrar)]
#![feature(box_syntax, rustc_private)]

// Load rustc as a plugin to get macros.
#[macro_use]
extern crate rustc;
extern crate rustc_driver;

use rustc::hir;
use rustc::lint::{LateContext, LintContext, LintPass, LateLintPass, LintArray, LintId};
use rustc_driver::plugin::Registry;

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
    reg.lint_store.register_lints(&[&TEST_LINT, &PLEASE_LINT]);
    reg.lint_store.register_late_pass(|| box Pass);
    reg.lint_store.register_group(true, "lint_me", None,
        vec![LintId::of(&TEST_LINT), LintId::of(&PLEASE_LINT)]);
}
