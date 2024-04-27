//@ build-fail

#![feature(repr_simd, intrinsics, rustc_attrs, adt_const_params)]
#![allow(incomplete_features)]

#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct i32x2(i32, i32);
#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct i32x4(i32, i32, i32, i32);
#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct i32x8(i32, i32, i32, i32,
             i32, i32, i32, i32);

#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct f32x2(f32, f32);
#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct f32x4(f32, f32, f32, f32);
#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct f32x8(f32, f32, f32, f32,
             f32, f32, f32, f32);

extern "rust-intrinsic" {
    fn simd_insert<T, E>(x: T, idx: u32, y: E) -> T;
    fn simd_extract<T, E>(x: T, idx: u32) -> E;

    fn simd_shuffle<T, I, U>(x: T, y: T, idx: I) -> U;
    fn simd_shuffle_generic<T, U, const IDX: &'static [u32]>(x: T, y: T) -> U;
}

fn main() {
    let x = i32x4(0, 0, 0, 0);

    unsafe {
        simd_insert(0, 0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_insert(x, 0, 1.0);
        //~^ ERROR expected inserted type `i32` (element of input `i32x4`), found `f64`
        simd_extract::<_, f32>(x, 0);
        //~^ ERROR expected return type `i32` (element of input `i32x4`), found `f32`

        const IDX2: [u32; 2] = [0; 2];
        simd_shuffle::<i32, _, i32>(0, 0, IDX2);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        const IDX4: [u32; 4] = [0; 4];
        simd_shuffle::<i32, _, i32>(0, 0, IDX4);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        const IDX8: [u32; 8] = [0; 8];
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
        simd_shuffle_generic::<i32, i32, I2>(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        const I4: &[u32] = &[0; 4];
        simd_shuffle_generic::<i32, i32, I4>(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        const I8: &[u32] = &[0; 8];
        simd_shuffle_generic::<i32, i32, I8>(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`

        simd_shuffle_generic::<_, f32x2, I2>(x, x);
//~^ ERROR element type `i32` (element of input `i32x4`), found `f32x2` with element type `f32`
        simd_shuffle_generic::<_, f32x4, I4>(x, x);
//~^ ERROR element type `i32` (element of input `i32x4`), found `f32x4` with element type `f32`
        simd_shuffle_generic::<_, f32x8, I8>(x, x);
//~^ ERROR element type `i32` (element of input `i32x4`), found `f32x8` with element type `f32`

        simd_shuffle_generic::<_, i32x8, I2>(x, x);
        //~^ ERROR expected return type of length 2, found `i32x8` with length 8
        simd_shuffle_generic::<_, i32x8, I4>(x, x);
        //~^ ERROR expected return type of length 4, found `i32x8` with length 8
        simd_shuffle_generic::<_, i32x2, I8>(x, x);
        //~^ ERROR expected return type of length 8, found `i32x2` with length 2
    }
}
