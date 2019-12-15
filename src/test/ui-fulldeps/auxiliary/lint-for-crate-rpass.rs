// force-host

#![feature(plugin_registrar, rustc_private)]
#![feature(box_syntax)]
#[macro_use] extern crate rustc;
#[macro_use] extern crate rustc_session;
extern crate rustc_driver;
extern crate syntax;

use rustc::lint::{LateContext, LintContext, LintPass, LateLintPass};
use rustc_driver::plugin::Registry;
use rustc::hir;
use syntax::attr;
use syntax::symbol::Symbol;

macro_rules! fake_lint_pass {
    ($struct:ident, $($attr:expr),*) => {
        struct $struct;

        impl LintPass for $struct {
            fn name(&self) -> &'static str {
                stringify!($struct)
            }
        }

        impl<'a, 'tcx> LateLintPass<'a, 'tcx> for $struct {
            fn check_crate(&mut self, cx: &LateContext, krate: &hir::Crate) {
                $(
                    if !attr::contains_name(&krate.attrs, $attr) {
                        cx.span_lint(CRATE_NOT_OKAY, krate.span,
                                     &format!("crate is not marked with #![{}]", $attr));
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
    Symbol::intern("rustc_crate_okay")
}

fake_lint_pass! {
    PassRedBlue,
    Symbol::intern("rustc_crate_red"), Symbol::intern("rustc_crate_blue")
}

fake_lint_pass! {
    PassGreyGreen,
    Symbol::intern("rustc_crate_grey"), Symbol::intern("rustc_crate_green")
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.lint_store.register_lints(&[
        &CRATE_NOT_OKAY,
        &CRATE_NOT_RED,
        &CRATE_NOT_BLUE,
        &CRATE_NOT_GREY,
        &CRATE_NOT_GREEN,
    ]);
    reg.lint_store.register_late_pass(|| box PassOkay);
    reg.lint_store.register_late_pass(|| box PassRedBlue);
    reg.lint_store.register_late_pass(|| box PassGreyGreen);
}
