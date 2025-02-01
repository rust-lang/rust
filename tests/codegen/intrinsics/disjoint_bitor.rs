//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::disjoint_bitor;

// CHECK-LABEL: @disjoint_bitor_signed
#[no_mangle]
pub unsafe fn disjoint_bitor_signed(x: i32, y: i32) -> i32 {
    // CHECK: or disjoint i32 %x, %y
    disjoint_bitor(x, y)
}

// CHECK-LABEL: @disjoint_bitor_unsigned
#[no_mangle]
pub unsafe fn disjoint_bitor_unsigned(x: u64, y: u64) -> u64 {
    // CHECK: or disjoint i64 %x, %y
    disjoint_bitor(x, y)
}
