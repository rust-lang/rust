//@ build-fail

#![feature(repr_simd, core_intrinsics)]

#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct i32x4([i32; 4]);
#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct i16x8([i16; 8]);

use std::intrinsics::simd::{simd_eq, simd_ge, simd_gt, simd_le, simd_lt, simd_ne};

fn main() {
    let x = i32x4([0, 0, 0, 0]);

    unsafe {
        simd_eq::<i32, i32>(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_ne::<i32, i32>(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_lt::<i32, i32>(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_le::<i32, i32>(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_gt::<i32, i32>(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_ge::<i32, i32>(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`

        simd_eq::<_, i32>(x, x);
        //~^ ERROR expected SIMD return type, found non-SIMD `i32`
        simd_ne::<_, i32>(x, x);
        //~^ ERROR expected SIMD return type, found non-SIMD `i32`
        simd_lt::<_, i32>(x, x);
        //~^ ERROR expected SIMD return type, found non-SIMD `i32`
        simd_le::<_, i32>(x, x);
        //~^ ERROR expected SIMD return type, found non-SIMD `i32`
        simd_gt::<_, i32>(x, x);
        //~^ ERROR expected SIMD return type, found non-SIMD `i32`
        simd_ge::<_, i32>(x, x);
        //~^ ERROR expected SIMD return type, found non-SIMD `i32`

        simd_eq::<_, i16x8>(x, x);
        //~^ ERROR return type with length 4 (same as input type `i32x4`), found `i16x8` with length 8
        simd_ne::<_, i16x8>(x, x);
        //~^ ERROR return type with length 4 (same as input type `i32x4`), found `i16x8` with length 8
        simd_lt::<_, i16x8>(x, x);
        //~^ ERROR return type with length 4 (same as input type `i32x4`), found `i16x8` with length 8
        simd_le::<_, i16x8>(x, x);
        //~^ ERROR return type with length 4 (same as input type `i32x4`), found `i16x8` with length 8
        simd_gt::<_, i16x8>(x, x);
        //~^ ERROR return type with length 4 (same as input type `i32x4`), found `i16x8` with length 8
        simd_ge::<_, i16x8>(x, x);
        //~^ ERROR return type with length 4 (same as input type `i32x4`), found `i16x8` with length 8
    }
}
