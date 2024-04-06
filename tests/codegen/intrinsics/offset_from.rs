//@ revisions: OPT1 OPT2
//@ [OPT1] compile-flags: -Copt-level=1
//@ [OPT2] compile-flags: -Copt-level=2
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
    // CHECK-NEXT: %[[D:.+]] = sub i64
    // CHECK-NOT: assume
    // CHECK-NEXT: sdiv exact i64 %[[D]], 3
    // CHECK-NEXT: ret i64
    ptr_offset_from(a, b)
}

// CHECK-LABEL: @offset_from_unsigned_odd_size
#[no_mangle]
pub unsafe fn offset_from_unsigned_odd_size(a: *const RGB, b: *const RGB) -> usize {
    // CHECK: start
    // CHECK-NEXT: ptrtoint
    // CHECK-NEXT: ptrtoint
    // CHECK-NEXT: %[[D:.+]] = sub nuw i64
    // OPT1-NOT: assume
    // OPT2-NEXT: %[[POS:.+]] = icmp sgt i64 %[[D]], -1
    // OPT2-NEXT: tail call void @llvm.assume(i1 %[[POS]])
    // CHECK-NEXT: udiv exact i64 %[[D]], 3
    // CHECK-NEXT: ret i64
    ptr_offset_from_unsigned(a, b)
}
