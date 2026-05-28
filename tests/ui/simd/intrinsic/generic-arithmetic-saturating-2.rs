//@ build-fail
//@ ignore-emscripten
#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

use std::intrinsics::simd::{simd_saturating_add, simd_saturating_sub};

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct i32x4(pub [i32; 4]);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct x4<T>(pub [T; 4]);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct f32x4(pub [f32; 4]);

fn main() {
    let x = i32x4([0, 0, 0, 0]);
    let y = x4([0_usize, 0, 0, 0]);
    let z = f32x4([0.0, 0.0, 0.0, 0.0]);

    unsafe {
        simd_saturating_add(x, x);
        simd_saturating_add(y, y);
        simd_saturating_sub(x, x);
        simd_saturating_sub(y, y);

        simd_saturating_add(z, z);
        //~^ ERROR expected element type `f32` of vector type `f32x4` to be a signed or unsigned integer type
        simd_saturating_sub(z, z);
        //~^ ERROR expected element type `f32` of vector type `f32x4` to be a signed or unsigned integer type
    }
}
