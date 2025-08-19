//@ compile-flags: -C opt-level=3 -C no-prepopulate-passes
//@ only-64bit

#![feature(core_intrinsics, repr_simd)]
#![no_std]
#![crate_type = "lib"]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use core::intrinsics::simd::{simd_add, simd_extract};

use minisimd::*;

#[repr(transparent)]
pub struct Transparent<T>(T);

// These tests don't actually care about the add/extract, but it ensures the
// aggregated temporaries are only used in potentially-SSA ways.

#[no_mangle]
pub fn simd_aggregate_pot(x: [u32; 4], y: [u32; 4]) -> u32 {
    // CHECK-LABEL: simd_aggregate_pot
    // CHECK: %a = load <4 x i32>, ptr %x, align 4
    // CHECK: %b = load <4 x i32>, ptr %y, align 4
    // CHECK: add <4 x i32> %a, %b

    unsafe {
        let a = Simd(x);
        let b = Simd(y);
        let c = simd_add(a, b);
        simd_extract(c, 1)
    }
}

#[no_mangle]
pub fn simd_aggregate_npot(x: [u32; 7], y: [u32; 7]) -> u32 {
    // CHECK-LABEL: simd_aggregate_npot
    // CHECK: %a = load <7 x i32>, ptr %x, align 4
    // CHECK: %b = load <7 x i32>, ptr %y, align 4
    // CHECK: add <7 x i32> %a, %b

    unsafe {
        let a = Simd(x);
        let b = Simd(y);
        let c = simd_add(a, b);
        simd_extract(c, 1)
    }
}

#[no_mangle]
pub fn packed_simd_aggregate_pot(x: [u32; 4], y: [u32; 4]) -> u32 {
    // CHECK-LABEL: packed_simd_aggregate_pot
    // CHECK: %a = load <4 x i32>, ptr %x, align 4
    // CHECK: %b = load <4 x i32>, ptr %y, align 4
    // CHECK: add <4 x i32> %a, %b

    unsafe {
        let a = PackedSimd(x);
        let b = PackedSimd(y);
        let c = simd_add(a, b);
        simd_extract(c, 1)
    }
}

#[no_mangle]
pub fn packed_simd_aggregate_npot(x: [u32; 7], y: [u32; 7]) -> u32 {
    // CHECK-LABEL: packed_simd_aggregate_npot
    // CHECK: %b = alloca [28 x i8], align 4
    // CHECK: %a = alloca [28 x i8], align 4
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a, ptr align 4 %x, i64 28, i1 false)
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %b, ptr align 4 %y, i64 28, i1 false)
    // CHECK: %[[TEMPA:.+]] = load <7 x i32>, ptr %a, align 4
    // CHECK: %[[TEMPB:.+]] = load <7 x i32>, ptr %b, align 4
    // CHECK: add <7 x i32> %[[TEMPA]], %[[TEMPB]]

    unsafe {
        let a = PackedSimd(x);
        let b = PackedSimd(y);
        let c = simd_add(a, b);
        simd_extract(c, 1)
    }
}

#[no_mangle]
pub fn transparent_simd_aggregate(x: [u32; 4]) -> u32 {
    // The transparent wrapper can just use the same SSA value as its field.
    // No extra processing or spilling needed.

    // CHECK-LABEL: transparent_simd_aggregate
    // CHECK-NOT: alloca
    // CHECK: %[[RET:.+]] = alloca [4 x i8]
    // CHECK-NOT: alloca
    // CHECK: %a = load <4 x i32>, ptr %x, align 4
    // CHECK: %[[TEMP:.+]] = extractelement <4 x i32> %a, i32 1
    // CHECK: store i32 %[[TEMP]], ptr %[[RET]]

    unsafe {
        let a = Simd(x);
        let b = Transparent(a);
        simd_extract(b.0, 1)
    }
}
