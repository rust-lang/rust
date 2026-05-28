//@ build-fail
//@ ignore-backends: gcc

#![feature(
    repr_simd,
    intrinsics,
    core_intrinsics,
    rustc_attrs,
    adt_const_params,
    unsized_const_params
)]
#![allow(incomplete_features)]

use std::intrinsics::simd::{simd_extract, simd_insert, simd_shuffle};

#[rustc_intrinsic]
unsafe fn simd_shuffle_const_generic<T, U, const IDX: &'static [u32]>(x: T, y: T) -> U;

#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct i32x2([i32; 2]);
#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct i32x4([i32; 4]);
#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct i32x8([i32; 8]);

#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct f32x2([f32; 2]);
#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct f32x4([f32; 4]);
#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct f32x8([f32; 8]);

#[repr(simd)]
struct SimdShuffleIdx<const LEN: usize>([u32; LEN]);

fn main() {
    let x = i32x4([0, 0, 0, 0]);

    unsafe {
        simd_insert(0, 0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_insert(x, 0, 1.0);
        //~^ ERROR expected inserted type `i32` (element of input `i32x4`), found `f64`
        simd_extract::<_, f32>(x, 0);
        //~^ ERROR expected return type `i32` (element of input `i32x4`), found `f32`

        const IDX2: SimdShuffleIdx<2> = SimdShuffleIdx([0; 2]);
        simd_shuffle::<i32, _, i32>(0, 0, IDX2);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        const IDX4: SimdShuffleIdx<4> = SimdShuffleIdx([0; 4]);
        simd_shuffle::<i32, _, i32>(0, 0, IDX4);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        const IDX8: SimdShuffleIdx<8> = SimdShuffleIdx([0; 8]);
        simd_shuffle::<i32, _, i32>(0, 0, IDX8);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`

        simd_shuffle::<_, _, f32x2>(x, x, IDX2);
        //~^ ERROR element type `i32` (element of input `i32x4`), found `f32x2` with element type `f32`
        simd_shuffle::<_, _, f32x4>(x, x, IDX4);
        //~^ ERROR element type `i32` (element of input `i32x4`), found `f32x4` with element type `f32`
        simd_shuffle::<_, _, f32x8>(x, x, IDX8);
        //~^ ERROR element type `i32` (element of input `i32x4`), found `f32x8` with element type `f32`

        simd_shuffle::<_, _, i32x8>(x, x, IDX2);
        //~^ ERROR expected return type of length 2, found `i32x8` with length 8
        simd_shuffle::<_, _, i32x8>(x, x, IDX4);
        //~^ ERROR expected return type of length 4, found `i32x8` with length 8
        simd_shuffle::<_, _, i32x2>(x, x, IDX8);
        //~^ ERROR expected return type of length 8, found `i32x2` with length 2

        const I2: &[u32] = &[0; 2];
        simd_shuffle_const_generic::<i32, i32, I2>(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        const I4: &[u32] = &[0; 4];
        simd_shuffle_const_generic::<i32, i32, I4>(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        const I8: &[u32] = &[0; 8];
        simd_shuffle_const_generic::<i32, i32, I8>(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`

        simd_shuffle_const_generic::<_, f32x2, I2>(x, x);
        //~^ ERROR element type `i32` (element of input `i32x4`), found `f32x2` with element type `f32`
        simd_shuffle_const_generic::<_, f32x4, I4>(x, x);
        //~^ ERROR element type `i32` (element of input `i32x4`), found `f32x4` with element type `f32`
        simd_shuffle_const_generic::<_, f32x8, I8>(x, x);
        //~^ ERROR element type `i32` (element of input `i32x4`), found `f32x8` with element type `f32`

        simd_shuffle_const_generic::<_, i32x8, I2>(x, x);
        //~^ ERROR expected return type of length 2, found `i32x8` with length 8
        simd_shuffle_const_generic::<_, i32x8, I4>(x, x);
        //~^ ERROR expected return type of length 4, found `i32x8` with length 8
        simd_shuffle_const_generic::<_, i32x2, I8>(x, x);
        //~^ ERROR expected return type of length 8, found `i32x2` with length 2
    }
}
