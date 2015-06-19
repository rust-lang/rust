// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C no-prepopulate-passes

// The expect intrinsic needs to have the output from the call fed directly
// into the branch, these tests make sure that happens

#![feature(core)]

use std::intrinsics::{likely,unlikely};

// CHECK-LABEL: direct_likely
#[no_mangle]
pub fn direct_likely(x: i32) {
    unsafe {
        // CHECK: [[VAR:%[0-9]+]] = call i1 @llvm.expect.i1(i1 %{{.*}}, i1 true)
        // CHECK: br i1 [[VAR]], label %{{.+}}, label %{{.*}}
        if likely(x == 1) {}
    }
}

// CHECK-LABEL: direct_unlikely
#[no_mangle]
pub fn direct_unlikely(x: i32) {
    unsafe {
        // CHECK: [[VAR:%[0-9]+]] = call i1 @llvm.expect.i1(i1 %{{.*}}, i1 false)
        // CHECK: br i1 [[VAR]], label %{{.+}}, label %{{.*}}
        if unlikely(x == 1) {}
    }
}

// CHECK-LABEL: wrapped_likely
// Make sure you can wrap just the call to the intrinsic in `unsafe` and still
// have it work
#[no_mangle]
pub fn wrapped_likely(x: i32) {
    // CHECK: [[VAR:%[0-9]+]] = call i1 @llvm.expect.i1(i1 %{{.*}}, i1 true)
    // CHECK: br i1 [[VAR]], label %{{.+}}, label %{{.*}}
    if unsafe { likely(x == 1) } {}
}

// CHECK-LABEL: wrapped_unlikely
#[no_mangle]
pub fn wrapped_unlikely(x: i32) {
    // CHECK: [[VAR:%[0-9]+]] = call i1 @llvm.expect.i1(i1 %{{.*}}, i1 false)
    // CHECK: br i1 [[VAR]], label %{{.+}}, label %{{.*}}
    if unsafe { unlikely(x == 1) } {}
}
