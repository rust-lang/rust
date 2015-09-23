// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host

#![feature(plugin_registrar)]
#![feature(box_syntax, rustc_private)]

extern crate rustc_front;

// Load rustc as a plugin to get macros
#[macro_use]
extern crate rustc;

use rustc_front::hir;
use rustc::lint::{LateContext, LintContext, LintPass, LateLintPass, LateLintPassObject, LintArray};
use rustc::plugin::Registry;

declare_lint!(TEST_LINT, Warn, "Warn about items named 'lintme'");

declare_lint!(PLEASE_LINT, Warn, "Warn about items named 'pleaselintme'");

struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(TEST_LINT, PLEASE_LINT)
    }
}

impl LateLintPass for Pass {
    fn check_item(&mut self, cx: &LateContext, it: &hir::Item) {
        match &*it.name.as_str() {
            "lintme" => cx.span_lint(TEST_LINT, it.span, "item is named 'lintme'"),
            "pleaselintme" => cx.span_lint(PLEASE_LINT, it.span, "item is named 'pleaselintme'"),
            _ => {}
        }
    }
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_late_lint_pass(box Pass as LateLintPassObject);
    reg.register_lint_group("lint_me", vec![TEST_LINT, PLEASE_LINT]);
}
