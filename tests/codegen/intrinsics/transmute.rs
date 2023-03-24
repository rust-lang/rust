// compile-flags: -O -C no-prepopulate-passes
// only-64bit (so I don't need to worry about usize)
// min-llvm-version: 15.0 # this test assumes `ptr`s

#![crate_type = "lib"]
#![feature(core_intrinsics)]
#![feature(custom_mir)]
#![feature(inline_const)]

use std::mem::transmute;

// Some of the cases here are statically rejected by `mem::transmute`, so
// we need to generate custom MIR for those cases to get to codegen.
use std::intrinsics::mir::*;

enum Never {}

#[repr(align(2))]
pub struct BigNever(Never, u16, Never);

#[repr(align(8))]
pub struct Scalar64(i64);

#[repr(C, align(4))]
pub struct Aggregate64(u16, u8, i8, f32);

// CHECK-LABEL: @check_bigger_size(
#[no_mangle]
#[custom_mir(dialect = "runtime", phase = "initial")]
pub unsafe fn check_bigger_size(x: u16) -> u32 {
    // CHECK: call void @llvm.trap
    mir!{
        {
            RET = CastTransmute(x);
            Return()
        }
    }
}

// CHECK-LABEL: @check_smaller_size(
#[no_mangle]
#[custom_mir(dialect = "runtime", phase = "initial")]
pub unsafe fn check_smaller_size(x: u32) -> u16 {
    // CHECK: call void @llvm.trap
    mir!{
        {
            RET = CastTransmute(x);
            Return()
        }
    }
}

// CHECK-LABEL: @check_to_uninhabited(
#[no_mangle]
#[custom_mir(dialect = "runtime", phase = "initial")]
pub unsafe fn check_to_uninhabited(x: u16) -> BigNever {
    // CHECK: call void @llvm.trap
    mir!{
        {
            RET = CastTransmute(x);
            Return()
        }
    }
}

// CHECK-LABEL: @check_from_uninhabited(
#[no_mangle]
#[custom_mir(dialect = "runtime", phase = "initial")]
pub unsafe fn check_from_uninhabited(x: BigNever) -> u16 {
    // CHECK: call void @llvm.trap
    mir!{
        {
            RET = CastTransmute(x);
            Return()
        }
    }
}

// CHECK-LABEL: @check_to_newtype(
#[no_mangle]
pub unsafe fn check_to_newtype(x: u64) -> Scalar64 {
    // CHECK: %0 = alloca i64
    // CHECK: store i64 %x, ptr %0
    // CHECK: %1 = load i64, ptr %0
    // CHECK: ret i64 %1
    transmute(x)
}

// CHECK-LABEL: @check_from_newtype(
#[no_mangle]
pub unsafe fn check_from_newtype(x: Scalar64) -> u64 {
    // CHECK: %0 = alloca i64
    // CHECK: store i64 %x, ptr %0
    // CHECK: %1 = load i64, ptr %0
    // CHECK: ret i64 %1
    transmute(x)
}

// CHECK-LABEL: @check_to_pair(
#[no_mangle]
pub unsafe fn check_to_pair(x: u64) -> Option<i32> {
    // CHECK: %0 = alloca { i32, i32 }, align 4
    // CHECK: store i64 %x, ptr %0, align 4
    transmute(x)
}

// CHECK-LABEL: @check_from_pair(
#[no_mangle]
pub unsafe fn check_from_pair(x: Option<i32>) -> u64 {
    // The two arguments are of types that are only 4-aligned, but they're
    // immediates so we can write using the destination alloca's alignment.
    const { assert!(std::mem::align_of::<Option<i32>>() == 4) };

    // CHECK: %0 = alloca i64, align 8
    // CHECK: store i32 %x.0, ptr %1, align 8
    // CHECK: store i32 %x.1, ptr %2, align 4
    // CHECK: %3 = load i64, ptr %0, align 8
    // CHECK: ret i64 %3
    transmute(x)
}

// CHECK-LABEL: @check_to_float(
#[no_mangle]
pub unsafe fn check_to_float(x: u32) -> f32 {
    // CHECK: %0 = alloca float
    // CHECK: store i32 %x, ptr %0
    // CHECK: %1 = load float, ptr %0
    // CHECK: ret float %1
    transmute(x)
}

// CHECK-LABEL: @check_from_float(
#[no_mangle]
pub unsafe fn check_from_float(x: f32) -> u32 {
    // CHECK: %0 = alloca i32
    // CHECK: store float %x, ptr %0
    // CHECK: %1 = load i32, ptr %0
    // CHECK: ret i32 %1
    transmute(x)
}

// CHECK-LABEL: @check_to_bytes(
#[no_mangle]
pub unsafe fn check_to_bytes(x: u32) -> [u8; 4] {
    // CHECK: %0 = alloca [4 x i8], align 1
    // CHECK: store i32 %x, ptr %0, align 1
    // CHECK: %1 = load i32, ptr %0, align 1
    // CHECK: ret i32 %1
    transmute(x)
}

// CHECK-LABEL: @check_from_bytes(
#[no_mangle]
pub unsafe fn check_from_bytes(x: [u8; 4]) -> u32 {
    // CHECK: %1 = alloca i32, align 4
    // CHECK: %x = alloca [4 x i8], align 1
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %1, ptr align 1 %x, i64 4, i1 false)
    // CHECK: %3 = load i32, ptr %1, align 4
    // CHECK: ret i32 %3
    transmute(x)
}

// CHECK-LABEL: @check_to_aggregate(
#[no_mangle]
pub unsafe fn check_to_aggregate(x: u64) -> Aggregate64 {
    // CHECK: %0 = alloca %Aggregate64, align 4
    // CHECK: store i64 %x, ptr %0, align 4
    // CHECK: %1 = load i64, ptr %0, align 4
    // CHECK: ret i64 %1
    transmute(x)
}

// CHECK-LABEL: @check_from_aggregate(
#[no_mangle]
pub unsafe fn check_from_aggregate(x: Aggregate64) -> u64 {
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %{{[0-9]+}}, ptr align 4 %x, i64 8, i1 false)
    transmute(x)
}

// CHECK-LABEL: @check_long_array_less_aligned(
#[no_mangle]
pub unsafe fn check_long_array_less_aligned(x: [u64; 100]) -> [u16; 400] {
    // CHECK-NEXT: start
    // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 2 %0, ptr align 8 %x, i64 800, i1 false)
    // CHECK-NEXT: ret void
    transmute(x)
}

// CHECK-LABEL: @check_long_array_more_aligned(
#[no_mangle]
pub unsafe fn check_long_array_more_aligned(x: [u8; 100]) -> [u32; 25] {
    // CHECK-NEXT: start
    // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %0, ptr align 1 %x, i64 100, i1 false)
    // CHECK-NEXT: ret void
    transmute(x)
}
