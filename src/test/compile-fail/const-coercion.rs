// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(const_fn)]

// Check that a non-const fn doesn't coerce to a const fn

fn foo(x: i32) -> i32 {
    x * 22
}

fn bar(x: fn(i32) -> i32) -> const fn(i32) -> i32 {
    x
    //~^ ERROR mismatched types
    //~| expected `const fn(i32) -> i32`
    //~| found `fn(i32) -> i32`
}

fn main() {
    let f = bar(foo);
    let x = f(2);
    assert_eq!(x, 44);
}
