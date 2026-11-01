//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes -Z mir-opt-level=0
//@ only-64bit (so I don't need to worry about usize)

#![crate_type = "lib"]
#![feature(core_intrinsics)]

// Here to have unit tests of what they actually emit and to track things like
// <https://github.com/rust-lang/rust/issues/152773>

use std::intrinsics::{align_of_val, size_of_val};

// CHECK-LABEL: @align_of_array(
#[no_mangle]
pub unsafe fn align_of_array(x: &[u16; 7]) -> usize {
    // CHECK: start:
    // CHECK-NEXT: ret i64 2
    align_of_val(x)
}

// CHECK-LABEL: @size_of_array(
#[no_mangle]
pub unsafe fn size_of_array(x: &[u16; 7]) -> usize {
    // CHECK: start:
    // CHECK-NEXT: ret i64 14
    size_of_val(x)
}

// CHECK-LABEL: @align_of_slice(
#[no_mangle]
pub unsafe fn align_of_slice(x: &[u16]) -> usize {
    // CHECK: start:
    // CHECK-NEXT: [[SIZE:%.+]] = mul nuw nsw i64 %x.1, 2
    // CHECK-NEXT: ret i64 2
    align_of_val(x)
}

// CHECK-LABEL: @size_of_slice(
#[no_mangle]
pub unsafe fn size_of_slice(x: &[u16]) -> usize {
    // CHECK: start:
    // CHECK-NEXT: [[SIZE:%.+]] = mul nuw nsw i64 %x.1, 2
    // CHECK-NEXT: ret i64 [[SIZE]]
    size_of_val(x)
}
