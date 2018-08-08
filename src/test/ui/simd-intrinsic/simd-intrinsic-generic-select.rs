// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the simd_select intrinsic produces ok-ish error
// messages when misused.

#![feature(repr_simd, platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct f32x4(pub f32, pub f32, pub f32, pub f32);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct u32x4(pub u32, pub u32, pub u32, pub u32);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq)]
struct b8x4(pub i8, pub i8, pub i8, pub i8);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq)]
struct b8x8(pub i8, pub i8, pub i8, pub i8,
            pub i8, pub i8, pub i8, pub i8);

extern "platform-intrinsic" {
    fn simd_select<T, U>(x: T, a: U, b: U) -> U;
}

fn main() {
    let m4 = b8x4(0, 0, 0, 0);
    let m8 = b8x8(0, 0, 0, 0, 0, 0, 0, 0);
    let x = u32x4(0, 0, 0, 0);
    let z = f32x4(0.0, 0.0, 0.0, 0.0);

    unsafe {
        simd_select(m4, x, x);

        simd_select(m8, x, x);
        //~^ ERROR mismatched lengths: mask length `8` != other vector length `4`

        simd_select(x, x, x);
        //~^ ERROR mask element type is `u32`, expected `i_`

        simd_select(z, z, z);
        //~^ ERROR mask element type is `f32`, expected `i_`
    }
}
