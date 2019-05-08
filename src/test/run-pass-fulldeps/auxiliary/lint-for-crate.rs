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

macro_rules! fake_lint_pass {
    ($struct:ident, $lints:expr, $($attr:expr),*) => {
        struct $struct;

        impl LintPass for $struct {
            fn name(&self) -> &'static str {
                stringify!($struct)
            }

            fn get_lints(&self) -> LintArray {
                $lints
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
    lint_array!(CRATE_NOT_OKAY), // Single lint
    Symbol::intern("rustc_crate_okay")
}

fake_lint_pass! {
    PassRedBlue,
    lint_array!(CRATE_NOT_RED, CRATE_NOT_BLUE), // Multiple lints
    Symbol::intern("rustc_crate_red"), Symbol::intern("rustc_crate_blue")
}

fake_lint_pass! {
    PassGreyGreen,
    lint_array!(CRATE_NOT_GREY, CRATE_NOT_GREEN, ), // Trailing comma
    Symbol::intern("rustc_crate_grey"), Symbol::intern("rustc_crate_green")
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_late_lint_pass(box PassOkay);
    reg.register_late_lint_pass(box PassRedBlue);
    reg.register_late_lint_pass(box PassGreyGreen);
}
