// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test no-special rooting is used for managed boxes

#![feature(managed_boxes)]

use std::gc::GC;

fn testfn(cond: bool) {
    let mut x = box(GC) 3;
    let mut y = box(GC) 4;

    let mut a = &*x;

    let mut exp = 3;
    if cond {
        a = &*y;

        exp = 4;
    }

    x = box(GC) 5; //~ERROR cannot assign to `x` because it is borrowed
    y = box(GC) 6; //~ERROR cannot assign to `y` because it is borrowed
    assert_eq!(*a, exp);
    assert_eq!(x, box(GC) 5);
    assert_eq!(y, box(GC) 6);
}

pub fn main() {}
