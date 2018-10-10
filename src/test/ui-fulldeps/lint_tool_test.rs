// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:lint_tool_test.rs
// ignore-stage1
// compile-flags: --cfg foo

#![feature(plugin)]
#![plugin(lint_tool_test)]
#![allow(dead_code)]
#![cfg_attr(foo, warn(test_lint))]
//~^ WARNING lint name `test_lint` is deprecated and may not have an effect in the future
//~^^ WARNING lint name `test_lint` is deprecated and may not have an effect in the future
#![deny(clippy_group)]
//~^ WARNING lint name `clippy_group` is deprecated and may not have an effect in the future

fn lintme() { } //~ ERROR item is named 'lintme'

#[allow(clippy::group)]
fn lintmetoo() {}

#[allow(clippy::test_lint)]
pub fn main() {
    fn lintme() { }
    fn lintmetoo() { } //~ ERROR item is named 'lintmetoo'
}

#[allow(test_group)]
//~^ WARNING lint name `test_group` is deprecated and may not have an effect in the future
#[deny(this_lint_does_not_exist)] //~ WARNING unknown lint: `this_lint_does_not_exist`
fn hello() {
    fn lintmetoo() { }
}
