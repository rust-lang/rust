//@ build-fail

#![feature(repr_simd, core_intrinsics)]

use std::intrinsics::simd::simd_cast;

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
struct f32x4([f32; 4]);
#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct f32x8([f32; 8]);

fn main() {
    let x = i32x4([0, 0, 0, 0]);

    unsafe {
        simd_cast::<i32, i32>(0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_cast::<i32, i32x4>(0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_cast::<i32x4, i32>(x);
        //~^ ERROR expected SIMD return type, found non-SIMD `i32`
        simd_cast::<_, i32x8>(x);
        //~^ ERROR return type with length 4 (same as input type `i32x4`), found `i32x8` with length 8
    }
}
