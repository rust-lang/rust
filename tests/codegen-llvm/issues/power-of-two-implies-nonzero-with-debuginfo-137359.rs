//! Regression test for <https://github.com/rust-lang/rust/issues/137359>.
//!
//! With debuginfo enabled, LLVM did not use the fact that a power of two
//! is never zero.

//@ compile-flags: -Copt-level=3 -Cdebuginfo=2
//@ min-llvm-version: 23

#![crate_type = "lib"]

use std::hint::assert_unchecked;

// CHECK-LABEL: @power_of_two_is_nonzero
#[no_mangle]
pub unsafe fn power_of_two_is_nonzero(x: u32) -> bool {
    // CHECK-NOT: icmp
    // CHECK: ret i1 false
    assert_unchecked(x.is_power_of_two());
    x == 0
}

// CHECK-LABEL: @rem_by_power_of_two
#[no_mangle]
pub unsafe fn rem_by_power_of_two(x: u32, y: u32) -> u32 {
    // CHECK-NOT: urem
    // CHECK-NOT: panic
    // CHECK: %[[MASK:.+]] = add i32 %y, -1
    // CHECK: %[[REM:.+]] = and i32 %[[MASK]], %x
    // CHECK: ret i32 %[[REM]]
    assert_unchecked(y.is_power_of_two());
    x % y
}
