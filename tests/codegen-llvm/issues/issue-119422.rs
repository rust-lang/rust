//! This test checks that compiler don't generate useless compares to zeros
//! for `NonZero` integer types.
//!
//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled
//@ edition: 2021
//@ only-64bit (because the LLVM type of i64 for usize shows up)
#![crate_type = "lib"]

use core::num::NonZero;
use core::ptr::NonNull;

// CHECK-LABEL: @check_non_null
#[no_mangle]
pub fn check_non_null(x: NonNull<u8>) -> bool {
    // CHECK: ret i1 false
    x.as_ptr().is_null()
}

// CHECK-LABEL: @equals_zero_is_false_u8
#[no_mangle]
pub fn equals_zero_is_false_u8(x: NonZero<u8>) -> bool {
    // CHECK-NOT: br
    // CHECK: ret i1 false
    // CHECK-NOT: br
    x.get() == 0
}

// CHECK-LABEL: @not_equals_zero_is_true_u8
#[no_mangle]
pub fn not_equals_zero_is_true_u8(x: NonZero<u8>) -> bool {
    // CHECK-NOT: br
    // CHECK: ret i1 true
    // CHECK-NOT: br
    x.get() != 0
}

// CHECK-LABEL: @equals_zero_is_false_i8
#[no_mangle]
pub fn equals_zero_is_false_i8(x: NonZero<i8>) -> bool {
    // CHECK-NOT: br
    // CHECK: ret i1 false
    // CHECK-NOT: br
    x.get() == 0
}

// CHECK-LABEL: @not_equals_zero_is_true_i8
#[no_mangle]
pub fn not_equals_zero_is_true_i8(x: NonZero<i8>) -> bool {
    // CHECK-NOT: br
    // CHECK: ret i1 true
    // CHECK-NOT: br
    x.get() != 0
}

// CHECK-LABEL: @usize_try_from_u32
#[no_mangle]
pub fn usize_try_from_u32(x: NonZero<u32>) -> NonZero<usize> {
    // CHECK-NOT: br
    // CHECK: zext i32 %{{.*}} to i64
    // CHECK-NOT: br
    // CHECK: ret i64
    x.try_into().unwrap()
}

// CHECK-LABEL: @isize_try_from_i32
#[no_mangle]
pub fn isize_try_from_i32(x: NonZero<i32>) -> NonZero<isize> {
    // CHECK-NOT: br
    // CHECK: sext i32 %{{.*}} to i64
    // CHECK-NOT: br
    // CHECK: ret i64
    x.try_into().unwrap()
}

// CHECK-LABEL: @u64_from_nonzero_is_not_zero
#[no_mangle]
pub fn u64_from_nonzero_is_not_zero(x: NonZero<u64>) -> bool {
    // CHECK-NOT: br
    // CHECK: ret i1 false
    // CHECK-NOT: br
    let v: u64 = x.into();
    v == 0
}
