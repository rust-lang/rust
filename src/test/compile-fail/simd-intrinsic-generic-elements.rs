// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(repr_simd, platform_intrinsics)]

#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct i32x2(i32, i32);
#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct i32x3(i32, i32, i32);
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
struct f32x3(f32, f32, f32);
#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct f32x4(f32, f32, f32, f32);
#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct f32x8(f32, f32, f32, f32,
             f32, f32, f32, f32);

extern "platform-intrinsic" {
    fn simd_insert<T, E>(x: T, idx: u32, y: E) -> T;
    fn simd_extract<T, E>(x: T, idx: u32) -> E;

    fn simd_shuffle2<T, U>(x: T, y: T, idx: [u32; 2]) -> U;
    fn simd_shuffle3<T, U>(x: T, y: T, idx: [u32; 3]) -> U;
    fn simd_shuffle4<T, U>(x: T, y: T, idx: [u32; 4]) -> U;
    fn simd_shuffle8<T, U>(x: T, y: T, idx: [u32; 8]) -> U;
}

fn main() {
    let x = i32x4(0, 0, 0, 0);

    unsafe {
        simd_insert(0, 0, 0);
        //~^ ERROR SIMD insert intrinsic monomorphized for non-SIMD input type
        simd_insert(x, 0, 1.0);
        //~^ ERROR SIMD insert intrinsic monomorphized with inserted type not SIMD element type
        simd_extract::<_, f32>(x, 0);
        //~^ ERROR SIMD insert intrinsic monomorphized with returned type not SIMD element type

        simd_shuffle2::<i32, i32>(0, 0, [0; 2]);
        //~^ ERROR SIMD shuffle intrinsic monomorphized with non-SIMD input type
        simd_shuffle3::<i32, i32>(0, 0, [0; 3]);
        //~^ ERROR SIMD shuffle intrinsic monomorphized with non-SIMD input type
        simd_shuffle4::<i32, i32>(0, 0, [0; 4]);
        //~^ ERROR SIMD shuffle intrinsic monomorphized with non-SIMD input type
        simd_shuffle8::<i32, i32>(0, 0, [0; 8]);
        //~^ ERROR SIMD shuffle intrinsic monomorphized with non-SIMD input type

        simd_shuffle2::<_, f32x2>(x, x, [0; 2]);
        //~^ ERROR SIMD shuffle intrinsic monomorphized with different input and return element
        simd_shuffle3::<_, f32x3>(x, x, [0; 3]);
        //~^ ERROR SIMD shuffle intrinsic monomorphized with different input and return element
        simd_shuffle4::<_, f32x4>(x, x, [0; 4]);
        //~^ ERROR SIMD shuffle intrinsic monomorphized with different input and return element
        simd_shuffle8::<_, f32x8>(x, x, [0; 8]);
        //~^ ERROR SIMD shuffle intrinsic monomorphized with different input and return element
    }
}
