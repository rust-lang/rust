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

// Issue #15750: a macro that internally parses its input and then
// uses `quote_expr!` to rearrange it should be hygiene-preserving.

#![feature(phase)]

#[phase(plugin)]
extern crate macro_crate_test;

fn main() {
    let x = 3i;
    assert_eq!(3, identity!(x));
    assert_eq!(6, identity!(x+x));
    let x = 4i;
    assert_eq!(4, identity!(x));
}
