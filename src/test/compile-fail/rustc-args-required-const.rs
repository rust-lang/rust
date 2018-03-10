// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(attr_literals, rustc_attrs, const_fn)]

#[rustc_args_required_const(0)]
fn foo(_a: i32) {
}

#[rustc_args_required_const(1)]
fn bar(_a: i32, _b: i32) {
}

const A: i32 = 3;

const fn baz() -> i32 {
    3
}

fn main() {
    foo(2);
    foo(2 + 3);
    foo(baz());
    let a = 4;
    foo(A);
    foo(a); //~ ERROR: argument 1 is required to be a constant
    bar(a, 3);
    bar(a, a); //~ ERROR: argument 2 is required to be a constant
}
