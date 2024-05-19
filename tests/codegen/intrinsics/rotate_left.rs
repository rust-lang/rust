//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::rotate_left;

// CHECK-LABEL: @rotate_left_u16
#[no_mangle]
pub unsafe fn rotate_left_u16(x: u16, shift: u32) -> u16 {
    // CHECK: %[[tmp:.*]] = trunc i32 %shift to i16
    // CHECK: call i16 @llvm.fshl.i16(i16 %x, i16 %x, i16 %[[tmp]])
    rotate_left(x, shift)
}

// CHECK-LABEL: @rotate_left_u32
#[no_mangle]
pub unsafe fn rotate_left_u32(x: u32, shift: u32) -> u32 {
    // CHECK-NOT: trunc
    // CHECK-NOT: zext
    // CHECK: call i32 @llvm.fshl.i32(i32 %x, i32 %x, i32 %shift)
    rotate_left(x, shift)
}

// CHECK-LABEL: @rotate_left_u64
#[no_mangle]
pub unsafe fn rotate_left_u64(x: u64, shift: u32) -> u64 {
    // CHECK: %[[tmp:.*]] = zext i32 %shift to i64
    // CHECK: call i64 @llvm.fshl.i64(i64 %x, i64 %x, i64 %[[tmp]])
    rotate_left(x, shift)
}
