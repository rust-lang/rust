//@compile-flags: -C opt-level=3 -C no-prepopulate-passes

#![feature(
    core_intrinsics,
    repr_simd,
    arm_target_feature,
    mips_target_feature,
    s390x_target_feature,
    riscv_target_feature
)]
#![no_std]
#![crate_type = "lib"]
#![allow(non_camel_case_types)]

// Test that `core::intrinsics::simd::{simd_extract_dyn, simd_insert_dyn}`
// lower to an LLVM extractelement or insertelement operation.

use core::intrinsics::simd::{simd_extract, simd_extract_dyn, simd_insert, simd_insert_dyn};

#[repr(simd)]
#[derive(Clone, Copy)]
pub struct u32x16([u32; 16]);

#[repr(simd)]
#[derive(Clone, Copy)]
pub struct i8x16([i8; 16]);

// CHECK-LABEL: dyn_simd_extract
// CHECK: extractelement <16 x i8> %[[TEMP:.+]], i32 %idx
#[no_mangle]
#[cfg_attr(target_family = "wasm", target_feature(enable = "simd128"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86", target_feature(enable = "sse"))]
#[cfg_attr(target_arch = "mips", target_feature(enable = "msa"))]
#[cfg_attr(target_arch = "s390x", target_feature(enable = "vector"))]
#[cfg_attr(target_arch = "riscv64", target_feature(enable = "v"))]
unsafe extern "C" fn dyn_simd_extract(x: i8x16, idx: u32) -> i8 {
    simd_extract_dyn(x, idx)
}

// CHECK-LABEL: literal_dyn_simd_extract
// CHECK: extractelement <16 x i8> %[[TEMP:.+]], i32 7
#[no_mangle]
#[cfg_attr(target_family = "wasm", target_feature(enable = "simd128"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86", target_feature(enable = "sse"))]
#[cfg_attr(target_arch = "mips", target_feature(enable = "msa"))]
#[cfg_attr(target_arch = "s390x", target_feature(enable = "vector"))]
#[cfg_attr(target_arch = "riscv64", target_feature(enable = "v"))]
unsafe extern "C" fn literal_dyn_simd_extract(x: i8x16) -> i8 {
    simd_extract_dyn(x, 7)
}

// CHECK-LABEL: const_dyn_simd_extract
// CHECK: extractelement <16 x i8> %[[TEMP:.+]], i32 7
#[no_mangle]
#[cfg_attr(target_family = "wasm", target_feature(enable = "simd128"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86", target_feature(enable = "sse"))]
#[cfg_attr(target_arch = "mips", target_feature(enable = "msa"))]
#[cfg_attr(target_arch = "s390x", target_feature(enable = "vector"))]
#[cfg_attr(target_arch = "riscv64", target_feature(enable = "v"))]
unsafe extern "C" fn const_dyn_simd_extract(x: i8x16) -> i8 {
    simd_extract_dyn(x, const { 3 + 4 })
}

// CHECK-LABEL: const_simd_extract
// CHECK: extractelement <16 x i8> %[[TEMP:.+]], i32 7
#[no_mangle]
#[cfg_attr(target_family = "wasm", target_feature(enable = "simd128"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86", target_feature(enable = "sse"))]
#[cfg_attr(target_arch = "mips", target_feature(enable = "msa"))]
#[cfg_attr(target_arch = "s390x", target_feature(enable = "vector"))]
#[cfg_attr(target_arch = "riscv64", target_feature(enable = "v"))]
unsafe extern "C" fn const_simd_extract(x: i8x16) -> i8 {
    simd_extract(x, const { 3 + 4 })
}

// CHECK-LABEL: dyn_simd_insert
// CHECK: insertelement <16 x i8> %[[TEMP:.+]], i8 %e, i32 %idx
#[no_mangle]
#[cfg_attr(target_family = "wasm", target_feature(enable = "simd128"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86", target_feature(enable = "sse"))]
#[cfg_attr(target_arch = "mips", target_feature(enable = "msa"))]
#[cfg_attr(target_arch = "s390x", target_feature(enable = "vector"))]
#[cfg_attr(target_arch = "riscv64", target_feature(enable = "v"))]
unsafe extern "C" fn dyn_simd_insert(x: i8x16, e: i8, idx: u32) -> i8x16 {
    simd_insert_dyn(x, idx, e)
}

// CHECK-LABEL: literal_dyn_simd_insert
// CHECK: insertelement <16 x i8> %[[TEMP:.+]], i8 %e, i32 7
#[no_mangle]
#[cfg_attr(target_family = "wasm", target_feature(enable = "simd128"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86", target_feature(enable = "sse"))]
#[cfg_attr(target_arch = "mips", target_feature(enable = "msa"))]
#[cfg_attr(target_arch = "s390x", target_feature(enable = "vector"))]
#[cfg_attr(target_arch = "riscv64", target_feature(enable = "v"))]
unsafe extern "C" fn literal_dyn_simd_insert(x: i8x16, e: i8) -> i8x16 {
    simd_insert_dyn(x, 7, e)
}

// CHECK-LABEL: const_dyn_simd_insert
// CHECK: insertelement <16 x i8> %[[TEMP:.+]], i8 %e, i32 7
#[no_mangle]
#[cfg_attr(target_family = "wasm", target_feature(enable = "simd128"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86", target_feature(enable = "sse"))]
#[cfg_attr(target_arch = "mips", target_feature(enable = "msa"))]
#[cfg_attr(target_arch = "s390x", target_feature(enable = "vector"))]
#[cfg_attr(target_arch = "riscv64", target_feature(enable = "v"))]
unsafe extern "C" fn const_dyn_simd_insert(x: i8x16, e: i8) -> i8x16 {
    simd_insert_dyn(x, const { 3 + 4 }, e)
}

// CHECK-LABEL: const_simd_insert
// CHECK: insertelement <16 x i8> %[[TEMP:.+]], i8 %e, i32 7
#[no_mangle]
#[cfg_attr(target_family = "wasm", target_feature(enable = "simd128"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86", target_feature(enable = "sse"))]
#[cfg_attr(target_arch = "mips", target_feature(enable = "msa"))]
#[cfg_attr(target_arch = "s390x", target_feature(enable = "vector"))]
#[cfg_attr(target_arch = "riscv64", target_feature(enable = "v"))]
unsafe extern "C" fn const_simd_insert(x: i8x16, e: i8) -> i8x16 {
    simd_insert(x, const { 3 + 4 }, e)
}
