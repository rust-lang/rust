// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:macro_crate_test.rs
// ignore-stage1
// ignore-android

#![feature(phase)]

#[phase(plugin)]
extern crate macro_crate_test;

fn main() {
    macro_crate_test::foo();
    //~^ ERROR failed to resolve. Use of undeclared module `macro_crate_test`
    //~^^ ERROR unresolved name `macro_crate_test::foo`.
}
