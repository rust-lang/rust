// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// exec-env:RUST_POISON_ON_FREE=1

#![feature(managed_boxes)]

use std::gc::GC;

fn testfn(cond: bool) {
    let mut x = box(GC) 3i;
    let mut y = box(GC) 4i;

    // borrow x and y
    let r_x = &*x;
    let r_y = &*y;
    let mut r = r_x;
    let mut exp = 3;

    if cond {
        r = r_y;
        exp = 4;
    }

    println!("*r = {}, exp = {}", *r, exp);
    assert_eq!(*r, exp);

    x = box(GC) 5i; //~ERROR cannot assign to `x` because it is borrowed
    y = box(GC) 6i; //~ERROR cannot assign to `y` because it is borrowed

    println!("*r = {}, exp = {}", *r, exp);
    assert_eq!(*r, exp);
    assert_eq!(x, box(GC) 5i);
    assert_eq!(y, box(GC) 6i);
}

pub fn main() {
    testfn(true);
    testfn(false);
}
