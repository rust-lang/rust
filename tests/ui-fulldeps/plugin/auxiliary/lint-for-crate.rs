// force-host

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_lint;
#[macro_use]
extern crate rustc_session;
extern crate rustc_ast;
extern crate rustc_span;

use rustc_ast::attr;
use rustc_driver::plugin::Registry;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_span::def_id::CRATE_DEF_ID;
use rustc_span::symbol::Symbol;

declare_lint! {
    CRATE_NOT_OKAY,
    Warn,
    "crate not marked with #![crate_okay]"
}

declare_lint_pass!(Pass => [CRATE_NOT_OKAY]);

impl<'tcx> LateLintPass<'tcx> for Pass {
    fn check_crate(&mut self, cx: &LateContext) {
        let attrs = cx.tcx.hir().attrs(rustc_hir::CRATE_HIR_ID);
        let span = cx.tcx.def_span(CRATE_DEF_ID);
        if !attr::contains_name(attrs, Symbol::intern("crate_okay")) {
            cx.lint(CRATE_NOT_OKAY, "crate is not marked with #![crate_okay]", |lint| {
                lint.set_span(span)
            });
        }
    }
}

#[no_mangle]
fn __rustc_plugin_registrar(reg: &mut Registry) {
    reg.lint_store.register_lints(&[&CRATE_NOT_OKAY]);
    reg.lint_store.register_late_pass(|_| Box::new(Pass));
}
