// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;

struct A { x: i32, y: f64 }

#[cfg(not(works))]
unsafe fn access(n:*mut A) -> (i32, f64) {
    let x : i32 = n.x; //~ no field `x` on type `*mut A`
                       //~| NOTE `n` is a native pointer; perhaps you need to deref with `(*n).x`
    let y : f64 = n.y; //~ no field `y` on type `*mut A`
                       //~| NOTE `n` is a native pointer; perhaps you need to deref with `(*n).y`
    (x, y)
}

#[cfg(works)]
unsafe fn access(n:*mut A) -> (i32, f64) {
    let x : i32 = (*n).x;
    let y : f64 = (*n).y;
    (x, y)
}

fn main() {
    let a :  A = A { x: 3, y: 3.14 };
    let p : &A = &a;
    let (x,y) = unsafe {
        let n : *mut A = mem::transmute(p);
        access(n)
    };
    println!("x: {}, y: {}", x, y);
}
