//@ build-fail
//@ ignore-emscripten

// Test that the simd_reduce_{op} intrinsics produce ok-ish error
// messages when misused.

#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

use std::intrinsics::simd::*;

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct f32x4(pub [f32; 4]);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct u32x4(pub [u32; 4]);

fn main() {
    let x = u32x4([0, 0, 0, 0]);
    let z = f32x4([0.0, 0.0, 0.0, 0.0]);

    unsafe {
        simd_reduce_add_ordered(z, 0);
        //~^ ERROR expected return type `f32` (element of input `f32x4`), found `i32`
        simd_reduce_mul_ordered(z, 1);
        //~^ ERROR expected return type `f32` (element of input `f32x4`), found `i32`

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
    }
}
