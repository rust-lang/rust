// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host

#![feature(plugin_registrar, rustc_private)]
#![feature(box_syntax)]
#![feature(macro_vis_matcher)]

#[macro_use] extern crate rustc;
extern crate rustc_plugin;
extern crate syntax;

use rustc::lint::{LateContext, LintContext, LintPass, LateLintPass, LateLintPassObject, LintArray};
use rustc_plugin::Registry;
use rustc::hir;
use syntax::attr;

macro_rules! fake_lint_pass {
    ($struct:ident, $lints:expr, $($attr:expr),*) => {
        struct $struct;

        impl LintPass for $struct {
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
    "crate_okay"
}

fake_lint_pass! {
    PassRedBlue,
    lint_array!(CRATE_NOT_RED, CRATE_NOT_BLUE), // Multiple lints
    "crate_red", "crate_blue"
}

fake_lint_pass! {
    PassGreyGreen,
    lint_array!(CRATE_NOT_GREY, CRATE_NOT_GREEN, ), // Trailing comma
    "crate_grey", "crate_green"
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_late_lint_pass(box PassOkay);
    reg.register_late_lint_pass(box PassRedBlue);
    reg.register_late_lint_pass(box PassGreyGreen);
}
