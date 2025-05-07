//@ compile-flags: -C no-prepopulate-passes

#![feature(core_intrinsics)]

use std::intrinsics::sqrtf32;

// CHECK: @llvm.sqrt.f32(float) #{{[0-9]*}}

fn main() {
    unsafe {
        sqrtf32(0.0f32);
    }
}
