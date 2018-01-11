// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(test)]
extern crate test;

fn foo(x: i32, y: i32) -> i64 {
    (x + y) as i64
}

#[inline(never)]
fn bar() {
    let _f = Box::new(0);
    // This call used to trigger an LLVM bug in opt-level z where the base
    // pointer gets corrupted, see issue #45034
    let y: fn(i32, i32) -> i64 = test::black_box(foo);
    test::black_box(y(1, 2));
}

fn main() {
    bar();
}
