// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

#![feature(repr_simd, platform_intrinsics)]
#[allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f32x4(pub f32, pub f32, pub f32, pub f32);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f32x8(f32, f32, f32, f32, f32, f32, f32, f32);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct b8x4(pub i8, pub i8, pub i8, pub i8);

extern "platform-intrinsic" {
    fn simd_select<T, U>(x: T, a: U, b: U) -> U;
    fn simd_select_bitmask<T, U>(x: T, a: U, b: U) -> U;
}

// CHECK-LABEL: @select
#[no_mangle]
pub unsafe fn select(m: b8x4, a: f32x4, b: f32x4) -> f32x4 {
    // CHECK: select <4 x i1>
    simd_select(m, a, b)
}

// CHECK-LABEL: @select_bitmask
#[no_mangle]
pub unsafe fn select_bitmask(m: i8, a: f32x8, b: f32x8) -> f32x8 {
    // CHECK: select <8 x i1>
    simd_select_bitmask(m, a, b)
}
