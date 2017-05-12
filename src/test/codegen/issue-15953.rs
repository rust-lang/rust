// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that llvm generates `memcpy` for moving a value
// inside a function and moving an argument.

struct Foo {
    x: Vec<i32>,
}

#[inline(never)]
#[no_mangle]
// CHECK: memcpy
fn interior(x: Vec<i32>) -> Vec<i32> {
    let Foo { x } = Foo { x: x };
    x
}

#[inline(never)]
#[no_mangle]
// CHECK: memcpy
fn exterior(x: Vec<i32>) -> Vec<i32> {
    x
}

fn main() {
    let x = interior(Vec::new());
    println!("{:?}", x);

    let x = exterior(Vec::new());
    println!("{:?}", x);
}
