//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::ctpop;

// CHECK-LABEL: @ctpop_u16
#[no_mangle]
pub unsafe fn ctpop_u16(x: u16) -> u32 {
    // CHECK: %[[tmp:.*]] = call i16 @llvm.ctpop.i16(i16 %x)
    // CHECK: zext i16 %[[tmp]] to i32
    ctpop(x)
}

// CHECK-LABEL: @ctpop_u32
#[no_mangle]
pub unsafe fn ctpop_u32(x: u32) -> u32 {
    // CHECK: call i32 @llvm.ctpop.i32(i32 %x)
    // CHECK-NOT: zext
    // CHECK-NOT: trunc
    ctpop(x)
}

// CHECK-LABEL: @ctpop_u64
#[no_mangle]
pub unsafe fn ctpop_u64(x: u64) -> u32 {
    // CHECK: %[[tmp:.*]] = call i64 @llvm.ctpop.i64(i64 %x)
    // CHECK: trunc i64 %[[tmp]] to i32
    ctpop(x)
}
