#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics;

// CHECK-LABEL: @nearbyintf32
#[no_mangle]
pub unsafe fn nearbyintf32(a: f32) -> f32 {
    // CHECK: llvm.nearbyint.f32
    intrinsics::nearbyintf32(a)
}

// CHECK-LABEL: @nearbyintf64
#[no_mangle]
pub unsafe fn nearbyintf64(a: f64) -> f64 {
    // CHECK: llvm.nearbyint.f64
    intrinsics::nearbyintf64(a)
}
