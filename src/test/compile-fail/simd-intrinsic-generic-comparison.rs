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
struct i32x4(i32, i32, i32, i32);
#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct i16x8(i16, i16, i16, i16,
             i16, i16, i16, i16);

extern "platform-intrinsic" {
    fn simd_eq<T, U>(x: T, y: T) -> U;
    fn simd_ne<T, U>(x: T, y: T) -> U;
    fn simd_lt<T, U>(x: T, y: T) -> U;
    fn simd_le<T, U>(x: T, y: T) -> U;
    fn simd_gt<T, U>(x: T, y: T) -> U;
    fn simd_ge<T, U>(x: T, y: T) -> U;
}

fn main() {
    let x = i32x4(0, 0, 0, 0);

    unsafe {
        simd_eq::<i32, i32>(0, 0);
        //~^ ERROR SIMD comparison intrinsic monomorphized for non-SIMD argument type
        simd_ne::<i32, i32>(0, 0);
        //~^ ERROR SIMD comparison intrinsic monomorphized for non-SIMD argument type
        simd_lt::<i32, i32>(0, 0);
        //~^ ERROR SIMD comparison intrinsic monomorphized for non-SIMD argument type
        simd_le::<i32, i32>(0, 0);
        //~^ ERROR SIMD comparison intrinsic monomorphized for non-SIMD argument type
        simd_gt::<i32, i32>(0, 0);
        //~^ ERROR SIMD comparison intrinsic monomorphized for non-SIMD argument type
        simd_ge::<i32, i32>(0, 0);
        //~^ ERROR SIMD comparison intrinsic monomorphized for non-SIMD argument type

        simd_eq::<_, i32>(x, x);
        //~^ ERROR SIMD comparison intrinsic monomorphized for non-SIMD return type
        simd_ne::<_, i32>(x, x);
        //~^ ERROR SIMD comparison intrinsic monomorphized for non-SIMD return type
        simd_lt::<_, i32>(x, x);
        //~^ ERROR SIMD comparison intrinsic monomorphized for non-SIMD return type
        simd_le::<_, i32>(x, x);
        //~^ ERROR SIMD comparison intrinsic monomorphized for non-SIMD return type
        simd_gt::<_, i32>(x, x);
        //~^ ERROR SIMD comparison intrinsic monomorphized for non-SIMD return type
        simd_ge::<_, i32>(x, x);
        //~^ ERROR SIMD comparison intrinsic monomorphized for non-SIMD return type

        simd_eq::<_, i16x8>(x, x);
//~^ ERROR monomorphized with input type `i32x4` and return type `i16x8` with different lengths
        simd_ne::<_, i16x8>(x, x);
//~^ ERROR monomorphized with input type `i32x4` and return type `i16x8` with different lengths
        simd_lt::<_, i16x8>(x, x);
//~^ ERROR monomorphized with input type `i32x4` and return type `i16x8` with different lengths
        simd_le::<_, i16x8>(x, x);
//~^ ERROR monomorphized with input type `i32x4` and return type `i16x8` with different lengths
        simd_gt::<_, i16x8>(x, x);
//~^ ERROR monomorphized with input type `i32x4` and return type `i16x8` with different lengths
        simd_ge::<_, i16x8>(x, x);
//~^ ERROR monomorphized with input type `i32x4` and return type `i16x8` with different lengths
    }
}
