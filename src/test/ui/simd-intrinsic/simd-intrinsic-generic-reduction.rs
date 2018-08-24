// ignore-emscripten

// Test that the simd_reduce_{op} intrinsics produce ok-ish error
// messages when misused.

#![feature(repr_simd, platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct f32x4(pub f32, pub f32, pub f32, pub f32);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct u32x4(pub u32, pub u32, pub u32, pub u32);


extern "platform-intrinsic" {
    fn simd_reduce_add_ordered<T, U>(x: T, y: U) -> U;
    fn simd_reduce_mul_ordered<T, U>(x: T, y: U) -> U;
    fn simd_reduce_and<T, U>(x: T) -> U;
    fn simd_reduce_or<T, U>(x: T) -> U;
    fn simd_reduce_xor<T, U>(x: T) -> U;
    fn simd_reduce_all<T>(x: T) -> bool;
    fn simd_reduce_any<T>(x: T) -> bool;
}

fn main() {
    let x = u32x4(0, 0, 0, 0);
    let z = f32x4(0.0, 0.0, 0.0, 0.0);

    unsafe {
        simd_reduce_add_ordered(z, 0_f32);
        simd_reduce_mul_ordered(z, 1_f32);

        simd_reduce_add_ordered(z, 2_f32);
        //~^ ERROR accumulator of simd_reduce_add_ordered is not 0.0
        simd_reduce_mul_ordered(z, 3_f32);
        //~^ ERROR accumulator of simd_reduce_mul_ordered is not 1.0

        let _: f32 = simd_reduce_and(x);
        //~^ ERROR expected return type `u32` (element of input `u32x4`), found `f32`
        let _: f32 = simd_reduce_or(x);
        //~^ ERROR expected return type `u32` (element of input `u32x4`), found `f32`
        let _: f32 = simd_reduce_xor(x);
        //~^ ERROR expected return type `u32` (element of input `u32x4`), found `f32`

        let _: f32 = simd_reduce_and(z);
        //~^ ERROR unsupported simd_reduce_and from `f32x4` with element `f32` to `f32`
        let _: f32 = simd_reduce_or(z);
        //~^ ERROR unsupported simd_reduce_or from `f32x4` with element `f32` to `f32`
        let _: f32 = simd_reduce_xor(z);
        //~^ ERROR unsupported simd_reduce_xor from `f32x4` with element `f32` to `f32`

        let _: bool = simd_reduce_all(z);
        //~^ ERROR unsupported simd_reduce_all from `f32x4` with element `f32` to `bool`
        let _: bool = simd_reduce_any(z);
        //~^ ERROR unsupported simd_reduce_any from `f32x4` with element `f32` to `bool`

        foo(0_f32);
    }
}

#[inline(never)]
unsafe fn foo(x: f32) {
    let z = f32x4(0.0, 0.0, 0.0, 0.0);
    simd_reduce_add_ordered(z, x);
    //~^ ERROR accumulator of simd_reduce_add_ordered is not a constant
    simd_reduce_mul_ordered(z, x);
    //~^ ERROR accumulator of simd_reduce_mul_ordered is not a constant
}
