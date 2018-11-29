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

#[macro_use] extern crate rustc;
extern crate rustc_plugin;
extern crate syntax;

use rustc::lint::{LateContext, LintContext, LintPass, LateLintPass, LateLintPassObject, LintArray};
use rustc_plugin::Registry;
use rustc::hir;
use syntax::attr;

declare_lint!(CRATE_NOT_OKAY, Warn, "crate not marked with #![crate_okay]");

struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(CRATE_NOT_OKAY)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_crate(&mut self, cx: &LateContext, krate: &hir::Crate) {
        if !attr::contains_name(&krate.attrs, "crate_okay") {
            cx.span_lint(CRATE_NOT_OKAY, krate.span,
                         "crate is not marked with #![crate_okay]");
        }
    }
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_late_lint_pass(box Pass);
}
