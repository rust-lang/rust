// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:lint_plugin_test.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(lint_plugin_test)]
#![forbid(test_lint)]
//~^ NOTE lint level defined here
//~| NOTE `forbid` level set here

fn lintme() { } //~ ERROR item is named 'lintme'

#[allow(test_lint)]
//~^ ERROR allow(test_lint) overruled by outer forbid(test_lint)
//~| NOTE overruled by previous forbid
pub fn main() {
    lintme();
}
