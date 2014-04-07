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
// force-host

// You'll note that there's lots of directives above. This is a very particular
// test in which we're both linking to a macro crate and loading macros from it.
// This implies that both versions are the host architecture, meaning this test
// must also be compiled with the host arch.
//
// because this doesn't work with that test runner, ignore-android because it
// can't run host binaries, and force-host to make this test build as the host
// arch.

#![feature(phase)]

#[phase(syntax, link)]
extern crate macro_crate_test;

fn main() {
    assert_eq!(1, make_a_1!());
    macro_crate_test::foo();
}
