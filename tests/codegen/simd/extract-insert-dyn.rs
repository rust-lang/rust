//@compile-flags: -Copt-level=3 -Z merge-functions=disabled

#![feature(core_intrinsics, s390x_target_feature, repr_simd)]
#![no_std]
#![crate_type = "lib"]
#![allow(non_camel_case_types)]

// test that `core::intrinsics::simd::{simd_extract_dyn, simd_insert_dyn}`
// optimize to their dedicated instructions on platforms that support them.

use core::intrinsics::simd::{simd_extract, simd_extract_dyn, simd_insert, simd_insert_dyn};

#[repr(simd)]
#[derive(Clone, Copy)]
pub struct u32x16([u32; 16]);

#[repr(simd)]
#[derive(Clone, Copy)]
pub struct i8x16([i8; 16]);

// CHECK-LABEL: dyn_simd_extract
// CHECK: extractelement <16 x i8> %x, i32 %idx
// CHECK-NEXT: ret
#[no_mangle]
unsafe extern "C" fn dyn_simd_extract(x: i8x16, idx: u32) -> i8 {
    simd_extract_dyn(x, idx)
}

// CHECK-LABEL: literal_dyn_simd_extract
// CHECK: extractelement <16 x i8> %x, i64 7
// CHECK-NEXT: ret
#[no_mangle]
unsafe extern "C" fn literal_dyn_simd_extract(x: i8x16) -> i8 {
    simd_extract_dyn(x, 7)
}

// CHECK-LABEL: const_dyn_simd_extract
// CHECK: extractelement <16 x i8> %x, i64 7
// CHECK-NEXT: ret
#[no_mangle]
unsafe extern "C" fn const_dyn_simd_extract(x: i8x16) -> i8 {
    simd_extract_dyn(x, const { 3 + 4 })
}

// CHECK-LABEL: const_simd_extract
// CHECK: extractelement <16 x i8> %x, i64 7
// CHECK-NEXT: ret
#[no_mangle]
unsafe extern "C" fn const_simd_extract(x: i8x16) -> i8 {
    simd_extract(x, const { 3 + 4 })
}

// CHECK-LABEL: dyn_simd_insert
// CHECK: insertelement <16 x i8> %x, i8 %e, i32 %idx
// CHECK-NEXT: ret
#[no_mangle]
unsafe extern "C" fn dyn_simd_insert(x: i8x16, e: i8, idx: u32) -> i8x16 {
    simd_insert_dyn(x, idx, e)
}

// CHECK-LABEL: literal_dyn_simd_insert
// CHECK: insertelement <16 x i8> %x, i8 %e, i64 7
// CHECK-NEXT: ret
#[no_mangle]
unsafe extern "C" fn literal_dyn_simd_insert(x: i8x16, e: i8) -> i8x16 {
    simd_insert_dyn(x, 7, e)
}

// CHECK-LABEL: const_dyn_simd_insert
// CHECK: insertelement <16 x i8> %x, i8 %e, i64 7
// CHECK-NEXT: ret
#[no_mangle]
unsafe extern "C" fn const_dyn_simd_insert(x: i8x16, e: i8) -> i8x16 {
    simd_insert_dyn(x, const { 3 + 4 }, e)
}

// CHECK-LABEL: const_simd_insert
// CHECK: insertelement <16 x i8> %x, i8 %e, i64 7
// CHECK-NEXT: ret
#[no_mangle]
unsafe extern "C" fn const_simd_insert(x: i8x16, e: i8) -> i8x16 {
    simd_insert(x, const { 3 + 4 }, e)
}
