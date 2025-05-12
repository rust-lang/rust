//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::cold_path;

#[no_mangle]
pub fn test_cold_path(x: bool) {
    cold_path();
}

// CHECK-LABEL: @test_cold_path(
// CHECK-NOT: cold_path
