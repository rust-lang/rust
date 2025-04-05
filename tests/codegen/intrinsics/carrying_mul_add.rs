//@ revisions: RAW OPT
//@ compile-flags: -C opt-level=1
//@[RAW] compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(core_intrinsics)]
#![feature(core_intrinsics_fallbacks)]

// Note that LLVM seems to sometimes permute the order of arguments to mul and add,
// so these tests don't check the arguments in the optimized revision.

use std::intrinsics::{carrying_mul_add, fallback};

// The fallbacks are emitted even when they're never used, but optimize out.

// RAW: wide_mul_u128
// OPT-NOT: wide_mul_u128

// CHECK-LABEL: @cma_u8
#[no_mangle]
pub unsafe fn cma_u8(a: u8, b: u8, c: u8, d: u8) -> (u8, u8) {
    // CHECK: [[A:%.+]] = zext i8 %a to i16
    // CHECK: [[B:%.+]] = zext i8 %b to i16
    // CHECK: [[C:%.+]] = zext i8 %c to i16
    // CHECK: [[D:%.+]] = zext i8 %d to i16
    // CHECK: [[AB:%.+]] = mul nuw i16
    // RAW-SAME: [[A]], [[B]]
    // CHECK: [[ABC:%.+]] = add nuw i16
    // RAW-SAME: [[AB]], [[C]]
    // CHECK: [[ABCD:%.+]] = add nuw i16
    // RAW-SAME: [[ABC]], [[D]]
    // CHECK: [[LOW:%.+]] = trunc i16 [[ABCD]] to i8
    // CHECK: [[HIGHW:%.+]] = lshr i16 [[ABCD]], 8
    // RAW: [[HIGH:%.+]] = trunc i16 [[HIGHW]] to i8
    // OPT: [[HIGH:%.+]] = trunc nuw i16 [[HIGHW]] to i8
    // CHECK: [[PAIR0:%.+]] = insertvalue { i8, i8 } poison, i8 [[LOW]], 0
    // CHECK: [[PAIR1:%.+]] = insertvalue { i8, i8 } [[PAIR0]], i8 [[HIGH]], 1
    // OPT: ret { i8, i8 } [[PAIR1]]
    carrying_mul_add(a, b, c, d)
}

// CHECK-LABEL: @cma_u32
#[no_mangle]
pub unsafe fn cma_u32(a: u32, b: u32, c: u32, d: u32) -> (u32, u32) {
    // CHECK: [[A:%.+]] = zext i32 %a to i64
    // CHECK: [[B:%.+]] = zext i32 %b to i64
    // CHECK: [[C:%.+]] = zext i32 %c to i64
    // CHECK: [[D:%.+]] = zext i32 %d to i64
    // CHECK: [[AB:%.+]] = mul nuw i64
    // RAW-SAME: [[A]], [[B]]
    // CHECK: [[ABC:%.+]] = add nuw i64
    // RAW-SAME: [[AB]], [[C]]
    // CHECK: [[ABCD:%.+]] = add nuw i64
    // RAW-SAME: [[ABC]], [[D]]
    // CHECK: [[LOW:%.+]] = trunc i64 [[ABCD]] to i32
    // CHECK: [[HIGHW:%.+]] = lshr i64 [[ABCD]], 32
    // RAW: [[HIGH:%.+]] = trunc i64 [[HIGHW]] to i32
    // OPT: [[HIGH:%.+]] = trunc nuw i64 [[HIGHW]] to i32
    // CHECK: [[PAIR0:%.+]] = insertvalue { i32, i32 } poison, i32 [[LOW]], 0
    // CHECK: [[PAIR1:%.+]] = insertvalue { i32, i32 } [[PAIR0]], i32 [[HIGH]], 1
    // OPT: ret { i32, i32 } [[PAIR1]]
    carrying_mul_add(a, b, c, d)
}

// CHECK-LABEL: @cma_u128
// CHECK-SAME: sret{{.+}}dereferenceable(32){{.+}}%_0,{{.+}}%a,{{.+}}%b,{{.+}}%c,{{.+}}%d
#[no_mangle]
pub unsafe fn cma_u128(a: u128, b: u128, c: u128, d: u128) -> (u128, u128) {
    // CHECK: [[A:%.+]] = zext i128 %a to i256
    // CHECK: [[B:%.+]] = zext i128 %b to i256
    // CHECK: [[C:%.+]] = zext i128 %c to i256
    // CHECK: [[D:%.+]] = zext i128 %d to i256
    // CHECK: [[AB:%.+]] = mul nuw i256
    // RAW-SAME: [[A]], [[B]]
    // CHECK: [[ABC:%.+]] = add nuw i256
    // RAW-SAME: [[AB]], [[C]]
    // CHECK: [[ABCD:%.+]] = add nuw i256
    // RAW-SAME: [[ABC]], [[D]]
    // CHECK: [[LOW:%.+]] = trunc i256 [[ABCD]] to i128
    // CHECK: [[HIGHW:%.+]] = lshr i256 [[ABCD]], 128
    // RAW: [[HIGH:%.+]] = trunc i256 [[HIGHW]] to i128
    // OPT: [[HIGH:%.+]] = trunc nuw i256 [[HIGHW]] to i128
    // RAW: [[PAIR0:%.+]] = insertvalue { i128, i128 } poison, i128 [[LOW]], 0
    // RAW: [[PAIR1:%.+]] = insertvalue { i128, i128 } [[PAIR0]], i128 [[HIGH]], 1
    // OPT: store i128 [[LOW]], ptr %_0
    // OPT: [[P1:%.+]] = getelementptr inbounds{{( nuw)?}} i8, ptr %_0, {{i32|i64}} 16
    // OPT: store i128 [[HIGH]], ptr [[P1]]
    // CHECK: ret void
    carrying_mul_add(a, b, c, d)
}

// CHECK-LABEL: @cma_i128
// CHECK-SAME: sret{{.+}}dereferenceable(32){{.+}}%_0,{{.+}}%a,{{.+}}%b,{{.+}}%c,{{.+}}%d
#[no_mangle]
pub unsafe fn cma_i128(a: i128, b: i128, c: i128, d: i128) -> (u128, i128) {
    // CHECK: [[A:%.+]] = sext i128 %a to i256
    // CHECK: [[B:%.+]] = sext i128 %b to i256
    // CHECK: [[C:%.+]] = sext i128 %c to i256
    // CHECK: [[D:%.+]] = sext i128 %d to i256
    // CHECK: [[AB:%.+]] = mul nsw i256
    // RAW-SAME: [[A]], [[B]]
    // CHECK: [[ABC:%.+]] = add nsw i256
    // RAW-SAME: [[AB]], [[C]]
    // CHECK: [[ABCD:%.+]] = add nsw i256
    // RAW-SAME: [[ABC]], [[D]]
    // CHECK: [[LOW:%.+]] = trunc i256 [[ABCD]] to i128
    // CHECK: [[HIGHW:%.+]] = lshr i256 [[ABCD]], 128
    // RAW: [[HIGH:%.+]] = trunc i256 [[HIGHW]] to i128
    // OPT: [[HIGH:%.+]] = trunc nuw i256 [[HIGHW]] to i128
    // RAW: [[PAIR0:%.+]] = insertvalue { i128, i128 } poison, i128 [[LOW]], 0
    // RAW: [[PAIR1:%.+]] = insertvalue { i128, i128 } [[PAIR0]], i128 [[HIGH]], 1
    // OPT: store i128 [[LOW]], ptr %_0
    // OPT: [[P1:%.+]] = getelementptr inbounds{{( nuw)?}} i8, ptr %_0, {{i32|i64}} 16
    // OPT: store i128 [[HIGH]], ptr [[P1]]
    // CHECK: ret void
    carrying_mul_add(a, b, c, d)
}

// CHECK-LABEL: @fallback_cma_u32
#[no_mangle]
pub unsafe fn fallback_cma_u32(a: u32, b: u32, c: u32, d: u32) -> (u32, u32) {
    // OPT-DAG: [[A:%.+]] = zext i32 %a to i64
    // OPT-DAG: [[B:%.+]] = zext i32 %b to i64
    // OPT-DAG: [[AB:%.+]] = mul nuw i64
    // OPT-DAG: [[C:%.+]] = zext i32 %c to i64
    // OPT-DAG: [[ABC:%.+]] = add nuw i64{{.+}}[[C]]
    // OPT-DAG: [[D:%.+]] = zext i32 %d to i64
    // OPT-DAG: [[ABCD:%.+]] = add nuw i64{{.+}}[[D]]
    // OPT-DAG: [[LOW:%.+]] = trunc i64 [[ABCD]] to i32
    // OPT-DAG: [[HIGHW:%.+]] = lshr i64 [[ABCD]], 32
    // OPT-DAG: [[HIGH:%.+]] = trunc nuw i64 [[HIGHW]] to i32
    // OPT-DAG: [[PAIR0:%.+]] = insertvalue { i32, i32 } poison, i32 [[LOW]], 0
    // OPT-DAG: [[PAIR1:%.+]] = insertvalue { i32, i32 } [[PAIR0]], i32 [[HIGH]], 1
    // OPT-DAG: ret { i32, i32 } [[PAIR1]]
    fallback::CarryingMulAdd::carrying_mul_add(a, b, c, d)
}
