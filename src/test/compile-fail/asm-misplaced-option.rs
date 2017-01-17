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
// ignore-arm
// ignore-aarch64
// ignore-s390x
// ignore-emscripten

#![feature(asm, rustc_attrs)]

#![allow(dead_code, non_upper_case_globals)]

#[cfg(any(target_arch = "x86",
          target_arch = "x86_64"))]
#[rustc_error]
pub fn main() { //~ ERROR compilation successful
    // assignment not dead
    let mut x: isize = 0;
    unsafe {
        // extra colon
        asm!("mov $1, $0" : "=r"(x) : "r"(5_usize), "0"(x) : : "cc");
        //~^ WARNING unrecognized option
    }
    assert_eq!(x, 5);

    unsafe {
        // comma in place of a colon
        asm!("add $2, $1; mov $1, $0" : "=r"(x) : "r"(x), "r"(8_usize) : "cc", "volatile");
        //~^ WARNING expected a clobber, found an option
    }
    assert_eq!(x, 13);
}
