// force-host

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_hir;
#[macro_use]
extern crate rustc_lint;
#[macro_use]
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_ast;

use rustc_driver::plugin::Registry;
use rustc_lint::{LateContext, LateLintPass, LintArray, LintContext, LintPass};
use rustc_span::symbol::Symbol;
use rustc_ast::attr;

declare_lint! {
    CRATE_NOT_OKAY,
    Warn,
    "crate not marked with #![crate_okay]"
}

declare_lint_pass!(Pass => [CRATE_NOT_OKAY]);

impl<'tcx> LateLintPass<'tcx> for Pass {
    fn check_crate(&mut self, cx: &LateContext, krate: &rustc_hir::Crate) {
        let attrs = cx.tcx.hir().attrs(rustc_hir::CRATE_HIR_ID);
        if !cx.sess().contains_name(attrs, Symbol::intern("crate_okay")) {
            cx.lint(CRATE_NOT_OKAY, |lint| {
                lint.build("crate is not marked with #![crate_okay]")
                    .set_span(krate.module().inner)
                    .emit()
            });
        }
    }
}

#[no_mangle]
fn __rustc_plugin_registrar(reg: &mut Registry) {
    reg.lint_store.register_lints(&[&CRATE_NOT_OKAY]);
    reg.lint_store.register_late_pass(|| Box::new(Pass));
}
