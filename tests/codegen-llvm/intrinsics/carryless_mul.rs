//@ compile-flags: -C opt-level=1

#![crate_type = "lib"]
#![feature(core_intrinsics)]
#![feature(uint_carryless_mul)]

use std::intrinsics::carryless_mul;

// CHECK-LABEL: @clmul_u8
#[no_mangle]
pub fn clmul_u8(a: u8, b: u8) -> u8 {
    // CHECK: [[RES:%.+]] = tail call i8 @llvm.clmul.i8(i8 %a, i8 %b)
    // CHECK: ret i8 [[RES]]
    carryless_mul(a, b)
}

// CHECK-LABEL: @clmul_u32
#[no_mangle]
pub fn clmul_u32(a: u32, b: u32) -> u32 {
    // CHECK: [[RES:%.+]] = tail call i32 @llvm.clmul.i32(i32 %a, i32 %b)
    // CHECK: ret i32 [[RES]]
    carryless_mul(a, b)
}

// CHECK-LABEL: @clmul_u128
#[no_mangle]
pub fn clmul_u128(a: u128, b: u128) -> u128 {
    // CHECK: [[RES:%.+]] = tail call i128 @llvm.clmul.i128(i128 %a, i128 %b)
    // CHECK: ret i128 [[RES]]
    carryless_mul(a, b)
}
