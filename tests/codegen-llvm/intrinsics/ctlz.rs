//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::{ctlz, ctlz_nonzero};

// CHECK-LABEL: @ctlz_u16
#[no_mangle]
pub unsafe fn ctlz_u16(x: u16) -> u32 {
    // CHECK: %[[tmp:.*]] = call i16 @llvm.ctlz.i16(i16 %x, i1 false)
    // CHECK: zext i16 %[[tmp]] to i32
    ctlz(x)
}

// CHECK-LABEL: @ctlz_nzu16
#[no_mangle]
pub unsafe fn ctlz_nzu16(x: u16) -> u32 {
    // CHECK: %[[tmp:.*]] = call i16 @llvm.ctlz.i16(i16 %x, i1 true)
    // CHECK: zext i16 %[[tmp]] to i32
    ctlz_nonzero(x)
}

// CHECK-LABEL: @ctlz_u32
#[no_mangle]
pub unsafe fn ctlz_u32(x: u32) -> u32 {
    // CHECK: call i32 @llvm.ctlz.i32(i32 %x, i1 false)
    // CHECK-NOT: zext
    // CHECK-NOT: trunc
    ctlz(x)
}

// CHECK-LABEL: @ctlz_nzu32
#[no_mangle]
pub unsafe fn ctlz_nzu32(x: u32) -> u32 {
    // CHECK: call i32 @llvm.ctlz.i32(i32 %x, i1 true)
    // CHECK-NOT: zext
    // CHECK-NOT: trunc
    ctlz_nonzero(x)
}

// CHECK-LABEL: @ctlz_u64
#[no_mangle]
pub unsafe fn ctlz_u64(x: u64) -> u32 {
    // CHECK: %[[tmp:.*]] = call i64 @llvm.ctlz.i64(i64 %x, i1 false)
    // CHECK: trunc i64 %[[tmp]] to i32
    ctlz(x)
}

// CHECK-LABEL: @ctlz_nzu64
#[no_mangle]
pub unsafe fn ctlz_nzu64(x: u64) -> u32 {
    // CHECK: %[[tmp:.*]] = call i64 @llvm.ctlz.i64(i64 %x, i1 true)
    // CHECK: trunc i64 %[[tmp]] to i32
    ctlz_nonzero(x)
}
