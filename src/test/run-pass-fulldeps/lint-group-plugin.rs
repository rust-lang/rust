// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:lint_group_plugin_test.rs
// ignore-stage1
#![feature(plugin)]
#![plugin(lint_group_plugin_test)]
#![allow(dead_code)]

fn lintme() { } //~ WARNING item is named 'lintme'
fn pleaselintme() { } //~ WARNING item is named 'pleaselintme'

#[allow(lint_me)]
pub fn main() {
    fn lintme() { }

    fn pleaselintme() { }
}
