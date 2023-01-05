// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

#![feature(repr_simd, platform_intrinsics, min_const_generics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct M(pub f32, pub f32, pub f32, pub f32);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct S<const N: usize>([f32; N]);

extern "platform-intrinsic" {
    fn simd_extract<T, U>(x: T, idx: u32) -> U;
    fn simd_insert<T, U>(x: T, idx: u32, b: U) -> T;
}

// CHECK-LABEL: @extract_m
#[no_mangle]
pub unsafe fn extract_m(v: M, i: u32) -> f32  {
    // CHECK: extractelement <4 x float> %{{v|_3}}, i32 %i
    simd_extract(v, i)
}

// CHECK-LABEL: @extract_s
#[no_mangle]
pub unsafe fn extract_s(v: S<4>, i: u32) -> f32  {
    // CHECK: extractelement <4 x float> %{{v|_3}}, i32 %i
    simd_extract(v, i)
}

// CHECK-LABEL: @insert_m
#[no_mangle]
pub unsafe fn insert_m(v: M, i: u32, j: f32) -> M  {
    // CHECK: insertelement <4 x float> %{{v|_4}}, float %j, i32 %i
    simd_insert(v, i, j)
}

// CHECK-LABEL: @insert_s
#[no_mangle]
pub unsafe fn insert_s(v: S<4>, i: u32, j: f32) -> S<4>  {
    // CHECK: insertelement <4 x float> %{{v|_4}}, float %j, i32 %i
    simd_insert(v, i, j)
}
