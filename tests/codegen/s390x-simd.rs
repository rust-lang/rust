//! test that s390x vector types are passed using `PassMode::Direct`
//! see also https://github.com/rust-lang/rust/issues/135744
//@ add-core-stubs
//@ compile-flags: --target s390x-unknown-linux-gnu -O
//@ needs-llvm-components: systemz

#![crate_type = "rlib"]
#![feature(no_core, asm_experimental_arch)]
#![feature(s390x_target_feature, simd_ffi, link_llvm_intrinsics, repr_simd)]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(simd)]
struct i8x16([i8; 16]);

#[repr(simd)]
struct i16x8([i16; 8]);

#[repr(simd)]
struct i32x4([i32; 4]);

#[repr(simd)]
struct i64x2([i64; 2]);

#[repr(simd)]
struct f32x4([f32; 4]);

#[repr(simd)]
struct f64x2([f64; 2]);

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.smax.v16i8"]
    fn vmxb(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.smax.v8i16"]
    fn vmxh(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.smax.v4i32"]
    fn vmxf(a: i32x4, b: i32x4) -> i32x4;
    #[link_name = "llvm.smax.v2i64"]
    fn vmxg(a: i64x2, b: i64x2) -> i64x2;
}

// CHECK-LABEL: define <16 x i8> @max_i8x16
// CHECK-SAME: <16 x i8> %a, <16 x i8> %b
// CHECK: call <16 x i8> @llvm.smax.v16i8(<16 x i8> %a, <16 x i8> %b)
#[no_mangle]
#[target_feature(enable = "vector")]
pub unsafe extern "C" fn max_i8x16(a: i8x16, b: i8x16) -> i8x16 {
    vmxb(a, b)
}

// CHECK-LABEL: define <8 x i16> @max_i16x8
// CHECK-SAME: <8 x i16> %a, <8 x i16> %b
// CHECK: call <8 x i16> @llvm.smax.v8i16(<8 x i16> %a, <8 x i16> %b)
#[no_mangle]
#[target_feature(enable = "vector")]
pub unsafe extern "C" fn max_i16x8(a: i16x8, b: i16x8) -> i16x8 {
    vmxh(a, b)
}

// CHECK-LABEL: define <4 x i32> @max_i32x4
// CHECK-SAME: <4 x i32> %a, <4 x i32> %b
// CHECK: call <4 x i32> @llvm.smax.v4i32(<4 x i32> %a, <4 x i32> %b)
#[no_mangle]
#[target_feature(enable = "vector")]
pub unsafe extern "C" fn max_i32x4(a: i32x4, b: i32x4) -> i32x4 {
    vmxf(a, b)
}

// CHECK-LABEL: define <2 x i64> @max_i64x2
// CHECK-SAME: <2 x i64> %a, <2 x i64> %b
// CHECK: call <2 x i64> @llvm.smax.v2i64(<2 x i64> %a, <2 x i64> %b)
#[no_mangle]
#[target_feature(enable = "vector")]
pub unsafe extern "C" fn max_i64x2(a: i64x2, b: i64x2) -> i64x2 {
    vmxg(a, b)
}

// CHECK-LABEL: define <4 x float> @choose_f32x4
// CHECK-SAME: <4 x float> %a, <4 x float> %b
#[no_mangle]
#[target_feature(enable = "vector")]
pub unsafe extern "C" fn choose_f32x4(a: f32x4, b: f32x4, c: bool) -> f32x4 {
    if c { a } else { b }
}

// CHECK-LABEL: define <2 x double> @choose_f64x2
// CHECK-SAME: <2 x double> %a, <2 x double> %b
#[no_mangle]
#[target_feature(enable = "vector")]
pub unsafe extern "C" fn choose_f64x2(a: f64x2, b: f64x2, c: bool) -> f64x2 {
    if c { a } else { b }
}

#[repr(C)]
struct Wrapper<T>(T);

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "vector")]
pub unsafe extern "C" fn max_wrapper_i8x16(a: Wrapper<i8x16>, b: Wrapper<i8x16>) -> Wrapper<i8x16> {
    // CHECK-LABEL: max_wrapper_i8x16
    // CHECK-SAME: sret([16 x i8])
    // CHECK-SAME: <16 x i8>
    // CHECK-SAME: <16 x i8>
    // CHECK: call <16 x i8> @llvm.smax.v16i8
    // CHECK-SAME: <16 x i8>
    // CHECK-SAME: <16 x i8>
    Wrapper(vmxb(a.0, b.0))
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "vector")]
pub unsafe extern "C" fn max_wrapper_i64x2(a: Wrapper<i64x2>, b: Wrapper<i64x2>) -> Wrapper<i64x2> {
    // CHECK-LABEL: max_wrapper_i64x2
    // CHECK-SAME: sret([16 x i8])
    // CHECK-SAME: <16 x i8>
    // CHECK-SAME: <16 x i8>
    // CHECK: call <2 x i64> @llvm.smax.v2i64
    // CHECK-SAME: <2 x i64>
    // CHECK-SAME: <2 x i64>
    Wrapper(vmxg(a.0, b.0))
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "vector")]
pub unsafe extern "C" fn choose_wrapper_f64x2(
    a: Wrapper<f64x2>,
    b: Wrapper<f64x2>,
    c: bool,
) -> Wrapper<f64x2> {
    // CHECK-LABEL: choose_wrapper_f64x2
    // CHECK-SAME: sret([16 x i8])
    // CHECK-SAME: <16 x i8>
    // CHECK-SAME: <16 x i8>
    Wrapper(choose_f64x2(a.0, b.0, c))
}

// CHECK: declare <2 x i64> @llvm.smax.v2i64(<2 x i64>, <2 x i64>)
