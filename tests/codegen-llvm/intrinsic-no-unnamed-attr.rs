//@ compile-flags: -C no-prepopulate-passes

#![feature(core_intrinsics)]

use std::intrinsics::sqrt;

// CHECK: @llvm.sqrt.f32(float) #{{[0-9]*}}

fn main() {
    sqrt(0.0f32);
}
