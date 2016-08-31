// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::{likely,unlikely};

#[no_mangle]
pub fn check_likely(x: i32, y: i32) -> Option<i32> {
    unsafe {
        // CHECK: call i1 @llvm.expect.i1(i1 %{{.*}}, i1 true)
        if likely(x == y) {
            None
        } else {
            Some(x + y)
        }
    }
}

#[no_mangle]
pub fn check_unlikely(x: i32, y: i32) -> Option<i32> {
    unsafe {
        // CHECK: call i1 @llvm.expect.i1(i1 %{{.*}}, i1 false)
        if unlikely(x == y) {
            None
        } else {
            Some(x + y)
        }
    }
}

