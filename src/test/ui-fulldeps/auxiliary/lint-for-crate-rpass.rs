// force-host

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_lint;
extern crate rustc_span;
#[macro_use]
extern crate rustc_session;
extern crate rustc_ast;

use rustc_ast::attr;
use rustc_driver::plugin::Registry;
use rustc_lint::{LateContext, LateLintPass, LintContext, LintPass};
use rustc_span::def_id::CRATE_DEF_ID;
use rustc_span::symbol::Symbol;

macro_rules! fake_lint_pass {
    ($struct:ident, $($attr:expr),*) => {
        struct $struct;

        impl LintPass for $struct {
            fn name(&self) -> &'static str {
                stringify!($struct)
            }
        }

        impl LateLintPass<'_> for $struct {
            fn check_crate(&mut self, cx: &LateContext) {
                let attrs = cx.tcx.hir().attrs(rustc_hir::CRATE_HIR_ID);
                let span = cx.tcx.def_span(CRATE_DEF_ID);
                $(
                    if !cx.sess().contains_name(attrs, $attr) {
                        cx.lint(CRATE_NOT_OKAY, |lint| {
                             let msg = format!("crate is not marked with #![{}]", $attr);
                             lint.build(&msg).set_span(span).emit()
                        });
                    }
                )*
            }
        }

    }
}

declare_lint!(CRATE_NOT_OKAY, Warn, "crate not marked with #![crate_okay]");
declare_lint!(CRATE_NOT_RED, Warn, "crate not marked with #![crate_red]");
declare_lint!(CRATE_NOT_BLUE, Warn, "crate not marked with #![crate_blue]");
declare_lint!(CRATE_NOT_GREY, Warn, "crate not marked with #![crate_grey]");
declare_lint!(CRATE_NOT_GREEN, Warn, "crate not marked with #![crate_green]");

fake_lint_pass! {
    PassOkay,
    Symbol::intern("crate_okay")
}

fake_lint_pass! {
    PassRedBlue,
    Symbol::intern("crate_red"), Symbol::intern("crate_blue")
}

fake_lint_pass! {
    PassGreyGreen,
    Symbol::intern("crate_grey"), Symbol::intern("crate_green")
}

#[no_mangle]
fn __rustc_plugin_registrar(reg: &mut Registry) {
    reg.lint_store.register_lints(&[
        &CRATE_NOT_OKAY,
        &CRATE_NOT_RED,
        &CRATE_NOT_BLUE,
        &CRATE_NOT_GREY,
        &CRATE_NOT_GREEN,
    ]);
    reg.lint_store.register_late_pass(|| Box::new(PassOkay));
    reg.lint_store.register_late_pass(|| Box::new(PassRedBlue));
    reg.lint_store.register_late_pass(|| Box::new(PassGreyGreen));
}
