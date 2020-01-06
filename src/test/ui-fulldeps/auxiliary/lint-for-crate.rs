// force-host

#![feature(plugin_registrar, rustc_private)]
#![feature(box_syntax)]

#[macro_use] extern crate rustc;
#[macro_use] extern crate rustc_session;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_span;
extern crate syntax;

use rustc::lint::{LateContext, LintContext, LintPass, LateLintPass, LintArray};
use rustc_driver::plugin::Registry;
use rustc_span::symbol::Symbol;
use syntax::attr;

declare_lint! {
    CRATE_NOT_OKAY,
    Warn,
    "crate not marked with #![crate_okay]"
}

declare_lint_pass!(Pass => [CRATE_NOT_OKAY]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_crate(&mut self, cx: &LateContext, krate: &rustc_hir::Crate) {
        if !attr::contains_name(&krate.attrs, Symbol::intern("crate_okay")) {
            cx.span_lint(CRATE_NOT_OKAY, krate.span,
                         "crate is not marked with #![crate_okay]");
        }
    }
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.lint_store.register_lints(&[&CRATE_NOT_OKAY]);
    reg.lint_store.register_late_pass(|| box Pass);
}
