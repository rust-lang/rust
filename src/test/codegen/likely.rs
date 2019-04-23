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
