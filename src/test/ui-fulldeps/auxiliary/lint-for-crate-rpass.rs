// force-host

#![feature(plugin_registrar, rustc_private)]
#![feature(box_syntax)]
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_span;
#[macro_use]
extern crate rustc_lint;
#[macro_use]
extern crate rustc_session;
extern crate rustc_ast;

use rustc_driver::plugin::Registry;
use rustc_lint::{LateContext, LateLintPass, LintContext, LintPass};
use rustc_span::symbol::Symbol;
use rustc_ast::attr;

macro_rules! fake_lint_pass {
    ($struct:ident, $($attr:expr),*) => {
        struct $struct;

        impl LintPass for $struct {
            fn name(&self) -> &'static str {
                stringify!($struct)
            }
        }

        impl<'a, 'tcx> LateLintPass<'a, 'tcx> for $struct {
            fn check_crate(&mut self, cx: &LateContext, krate: &rustc_hir::Crate) {
                $(
                    if !attr::contains_name(&krate.item.attrs, $attr) {
                        cx.lint(CRATE_NOT_OKAY, |lint| {
                             let msg = format!("crate is not marked with #![{}]", $attr);
                             lint.build(&msg).set_span(krate.item.span).emit()
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
