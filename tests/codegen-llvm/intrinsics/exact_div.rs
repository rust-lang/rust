//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::exact_div;

// CHECK-LABEL: @exact_sdiv
#[no_mangle]
pub unsafe fn exact_sdiv(x: i32, y: i32) -> i32 {
    // CHECK: sdiv exact
    exact_div(x, y)
}

// CHECK-LABEL: @exact_udiv
#[no_mangle]
pub unsafe fn exact_udiv(x: u32, y: u32) -> u32 {
    // CHECK: udiv exact
    exact_div(x, y)
}
