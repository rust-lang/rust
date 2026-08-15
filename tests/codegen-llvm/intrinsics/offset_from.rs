//@ compile-flags: -C opt-level=1
//@ only-64bit (because we're using [ui]size)

#![crate_type = "lib"]
#![feature(core_intrinsics)]

//! Basic optimizations are enabled because otherwise `x86_64-gnu-nopt` had an alloca.
//! Uses a type with non-power-of-two size to avoid normalizations to shifts.

use std::intrinsics::*;

type RGB = [u8; 3];

// CHECK-LABEL: @offset_from_odd_size
#[no_mangle]
pub unsafe fn offset_from_odd_size(a: *const RGB, b: *const RGB) -> isize {
    // CHECK: start
    // CHECK-NEXT: ptrtoint
    // CHECK-NEXT: ptrtoint
    // CHECK-NEXT: sub i64
    // CHECK-NEXT: sdiv exact i64 %{{[0-9]+}}, 3
    // CHECK-NEXT: ret i64
    ptr_offset_from(a, b)
}

// CHECK-LABEL: @offset_from_unsigned_odd_size
#[no_mangle]
pub unsafe fn offset_from_unsigned_odd_size(a: *const RGB, b: *const RGB) -> usize {
    // CHECK: start
    // CHECK-NEXT: ptrtoint
    // CHECK-NEXT: ptrtoint
    // CHECK-NEXT: sub nuw i64
    // CHECK-NEXT: udiv exact i64 %{{[0-9]+}}, 3
    // CHECK-NEXT: ret i64
    ptr_offset_from_unsigned(a, b)
}
