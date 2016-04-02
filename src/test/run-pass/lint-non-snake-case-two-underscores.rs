// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(non_snake_case)]
#![allow(dead_code)]

// These tests should not fail because they all begin with '__'.

struct Foo {
    __FooVariable: i32,
    __cOoKiE: f64
}

impl Foo {
    fn __TestMethod() { }
    fn __nOn____snake__CASE() { }
}

trait X {
    fn __ABC() { }
    fn __X_y__ZZ() { }
}

impl X for Foo {
    fn __ABC() { }
    fn __X_y__ZZ() { }
}

fn __BISCUIT() { }
fn __THIs_____iSNoT____snaKE____caSE() { }

fn main() {
    let __barXYZ: i32 = 0;
    let __Foo22 = "str";
}
