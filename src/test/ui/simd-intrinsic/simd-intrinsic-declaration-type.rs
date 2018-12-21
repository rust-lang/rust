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
struct i16x8(i16, i16, i16, i16, i16, i16, i16, i16);
#[repr(simd)]
struct u16x8(u16, u16, u16, u16, u16, u16, u16, u16);

#[repr(simd)]
struct i8x16(i8, i8, i8, i8, i8, i8, i8, i8,
             i8, i8, i8, i8, i8, i8, i8, i8);
#[repr(simd)]
struct i32x4(i32, i32, i32, i32);
#[repr(simd)]
struct i32x8(i32, i32, i32, i32, i32, i32, i32, i32);
#[repr(simd)]
struct u32x4(u32, u32, u32, u32);
#[repr(simd)]
struct f32x4(f32, f32, f32, f32);
#[repr(simd)]
struct i64x2(i64, i64);

mod signedness {
    // signedness matters
    extern "platform-intrinsic" {
        fn aarch64_vld2q_dup_s32(x: *const u32) -> (::u32x4, ::u32x4);
        //~^ ERROR intrinsic argument 1 has wrong type
        //~| ERROR intrinsic return value has wrong type
    }
}

mod lengths {
    // as do lengths
    extern "platform-intrinsic" {
        fn aarch64_vld2q_dup_s32(x: *const i32) -> (::i32x8, ::i32x4);
        //~^ ERROR intrinsic return value has wrong type
    }
}


mod float_and_int {
    // and so does int vs. float:
    extern "platform-intrinsic" {
        fn aarch64_vld2q_dup_s32(x: *const i32) -> (::f32x4, ::f32x4);
        //~^ ERROR intrinsic return value has wrong type
    }
}

fn main() {}
