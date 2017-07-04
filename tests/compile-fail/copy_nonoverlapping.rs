// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(core_intrinsics)]

use std::intrinsics::*;

//error-pattern: copy_nonoverlapping called on overlapping ranges

fn main() {
    let mut data = [0u8; 16];
    unsafe {
        let a = &data[0] as *const _;
        let b = &mut data[1] as *mut _;
        std::ptr::copy_nonoverlapping(a, b, 2);
    }
}
