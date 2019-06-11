#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::*;

// CHECK-LABEL: @unchecked_add_signed
#[no_mangle]
pub unsafe fn unchecked_add_signed(a: i32, b: i32) -> i32 {
    // CHECK: add nsw
    unchecked_add(a, b)
}

// CHECK-LABEL: @unchecked_add_unsigned
#[no_mangle]
pub unsafe fn unchecked_add_unsigned(a: u32, b: u32) -> u32 {
    // CHECK: add nuw
    unchecked_add(a, b)
}

// CHECK-LABEL: @unchecked_sub_signed
#[no_mangle]
pub unsafe fn unchecked_sub_signed(a: i32, b: i32) -> i32 {
    // CHECK: sub nsw
    unchecked_sub(a, b)
}

// CHECK-LABEL: @unchecked_sub_unsigned
#[no_mangle]
pub unsafe fn unchecked_sub_unsigned(a: u32, b: u32) -> u32 {
    // CHECK: sub nuw
    unchecked_sub(a, b)
}

// CHECK-LABEL: @unchecked_mul_signed
#[no_mangle]
pub unsafe fn unchecked_mul_signed(a: i32, b: i32) -> i32 {
    // CHECK: mul nsw
    unchecked_mul(a, b)
}

// CHECK-LABEL: @unchecked_mul_unsigned
#[no_mangle]
pub unsafe fn unchecked_mul_unsigned(a: u32, b: u32) -> u32 {
    // CHECK: mul nuw
    unchecked_mul(a, b)
}
