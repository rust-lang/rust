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
#![allow(non_camel_case_types)]
#[repr(simd)]
#[derive(Copy, Clone)]
pub struct i32x4(pub i32, pub i32, pub i32, pub i32);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct u32x4(pub u32, pub u32, pub u32, pub u32);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct f32x4(pub f32, pub f32, pub f32, pub f32);

extern "platform-intrinsic" {
    fn simd_add<T>(x: T, y: T) -> T;
    fn simd_sub<T>(x: T, y: T) -> T;
    fn simd_mul<T>(x: T, y: T) -> T;
    fn simd_div<T>(x: T, y: T) -> T;
    fn simd_shl<T>(x: T, y: T) -> T;
    fn simd_shr<T>(x: T, y: T) -> T;
    fn simd_and<T>(x: T, y: T) -> T;
    fn simd_or<T>(x: T, y: T) -> T;
    fn simd_xor<T>(x: T, y: T) -> T;
}

fn main() {
    let x = i32x4(0, 0, 0, 0);
    let y = u32x4(0, 0, 0, 0);
    let z = f32x4(0.0, 0.0, 0.0, 0.0);

    unsafe {
        simd_add(x, x);
        simd_add(y, y);
        simd_add(z, z);
        simd_sub(x, x);
        simd_sub(y, y);
        simd_sub(z, z);
        simd_mul(x, x);
        simd_mul(y, y);
        simd_mul(z, z);

        simd_div(z, z);

        simd_shl(x, x);
        simd_shl(y, y);
        simd_shr(x, x);
        simd_shr(y, y);
        simd_and(x, x);
        simd_and(y, y);
        simd_or(x, x);
        simd_or(y, y);
        simd_xor(x, x);
        simd_xor(y, y);


        simd_add(0, 0);
        //~^ ERROR `simd_add` intrinsic monomorphized with non-SIMD type
        simd_sub(0, 0);
        //~^ ERROR `simd_sub` intrinsic monomorphized with non-SIMD type
        simd_mul(0, 0);
        //~^ ERROR `simd_mul` intrinsic monomorphized with non-SIMD type
        simd_div(0, 0);
        //~^ ERROR `simd_div` intrinsic monomorphized with non-SIMD type
        simd_shl(0, 0);
        //~^ ERROR `simd_shl` intrinsic monomorphized with non-SIMD type
        simd_shr(0, 0);
        //~^ ERROR `simd_shr` intrinsic monomorphized with non-SIMD type
        simd_and(0, 0);
        //~^ ERROR `simd_and` intrinsic monomorphized with non-SIMD type
        simd_or(0, 0);
        //~^ ERROR `simd_or` intrinsic monomorphized with non-SIMD type
        simd_xor(0, 0);
        //~^ ERROR `simd_xor` intrinsic monomorphized with non-SIMD type


        simd_div(x, x);
//~^ ERROR `simd_div` intrinsic monomorphized with SIMD vector `i32x4` with unsupported element type
        simd_div(y, y);
//~^ ERROR `simd_div` intrinsic monomorphized with SIMD vector `u32x4` with unsupported element type
        simd_shl(z, z);
//~^ ERROR `simd_shl` intrinsic monomorphized with SIMD vector `f32x4` with unsupported element type
        simd_shr(z, z);
//~^ ERROR `simd_shr` intrinsic monomorphized with SIMD vector `f32x4` with unsupported element type
        simd_and(z, z);
//~^ ERROR `simd_and` intrinsic monomorphized with SIMD vector `f32x4` with unsupported element type
        simd_or(z, z);
//~^ ERROR `simd_or` intrinsic monomorphized with SIMD vector `f32x4` with unsupported element type
        simd_xor(z, z);
//~^ ERROR `simd_xor` intrinsic monomorphized with SIMD vector `f32x4` with unsupported element type
    }
}
