// force-host

#![feature(plugin_registrar, rustc_private)]
#![feature(box_syntax)]

#[macro_use] extern crate rustc;
extern crate rustc_plugin;
extern crate syntax;

use rustc::lint::{LateContext, LintContext, LintPass, LateLintPass, LateLintPassObject, LintArray};
use rustc_plugin::Registry;
use rustc::hir;
use syntax::attr;
use syntax::symbol::Symbol;

declare_lint! {
    CRATE_NOT_OKAY,
    Warn,
    "crate not marked with #![crate_okay]"
}

declare_lint_pass!(Pass => [CRATE_NOT_OKAY]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_crate(&mut self, cx: &LateContext, krate: &hir::Crate) {
        if !attr::contains_name(&krate.attrs, Symbol::intern("crate_okay")) {
            cx.span_lint(CRATE_NOT_OKAY, krate.span,
                         "crate is not marked with #![crate_okay]");
        }
    }
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_late_lint_pass(box Pass);
}
