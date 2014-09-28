// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android

#![feature(asm)]

#![allow(dead_code)]

#[cfg(any(target_arch = "x86",
          target_arch = "x86_64"))]
pub fn main() {
    // assignment not dead
    let mut x: int = 0;
    unsafe {
        // extra colon
        asm!("mov $1, $0" : "=r"(x) : "r"(5u), "0"(x) : : "cc");
        //~^ WARNING unrecognized option
    }
    assert_eq!(x, 5);

    unsafe {
        // comma in place of a colon
        asm!("add $2, $1; mov $1, $0" : "=r"(x) : "r"(x), "r"(8u) : "cc", "volatile");
        //~^ WARNING expected a clobber, found an option
    }
    assert_eq!(x, 13);
}

// At least one error is needed so that compilation fails
#[static_assert]
static b: bool = false; //~ ERROR static assertion failed
