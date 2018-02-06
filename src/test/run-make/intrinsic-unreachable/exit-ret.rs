// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(asm)]
#![crate_type="lib"]

#[deny(unreachable_code)]
pub fn exit(n: usize) -> i32 {
    unsafe {
        // Pretend this asm is an exit() syscall.
        asm!("" :: "r"(n) :: "volatile");
        // Can't actually reach this point, but rustc doesn't know that.
    }
    // This return value is just here to generate some extra code for a return
    // value, making it easier for the test script to detect whether the
    // compiler deleted it.
    42
}
