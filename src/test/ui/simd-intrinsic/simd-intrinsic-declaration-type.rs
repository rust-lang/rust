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
struct i16x8(i16, i16, i16, i16, i16, i16, i16, i16);
#[repr(simd)]
struct u16x8(u16, u16, u16, u16, u16, u16, u16, u16);

#[repr(simd)]
struct i8x16(i8, i8, i8, i8, i8, i8, i8, i8,
             i8, i8, i8, i8, i8, i8, i8, i8);
#[repr(simd)]
struct i32x4(i32, i32, i32, i32);
#[repr(simd)]
struct f32x4(f32, f32, f32, f32);
#[repr(simd)]
struct i64x2(i64, i64);

// correct signatures work well
mod right {
    use {i16x8, u16x8};
    extern "platform-intrinsic" {
        fn x86_mm_adds_epi16(x: i16x8, y: i16x8) -> i16x8;
        fn x86_mm_adds_epu16(x: u16x8, y: u16x8) -> u16x8;
    }
}
// but incorrect ones don't.

mod signedness {
    use {i16x8, u16x8};
    // signedness matters
    extern "platform-intrinsic" {
        fn x86_mm_adds_epi16(x: u16x8, y: u16x8) -> u16x8;
        //~^ ERROR intrinsic argument 1 has wrong type
        //~^^ ERROR intrinsic argument 2 has wrong type
        //~^^^ ERROR intrinsic return value has wrong type
        fn x86_mm_adds_epu16(x: i16x8, y: i16x8) -> i16x8;
        //~^ ERROR intrinsic argument 1 has wrong type
        //~^^ ERROR intrinsic argument 2 has wrong type
        //~^^^ ERROR intrinsic return value has wrong type
    }
}
// as do lengths
extern "platform-intrinsic" {
    fn x86_mm_adds_epi16(x: i8x16, y: i32x4) -> i64x2;
    //~^ ERROR intrinsic argument 1 has wrong type
    //~^^ ERROR intrinsic argument 2 has wrong type
    //~^^^ ERROR intrinsic return value has wrong type
}
// and so does int vs. float:
extern "platform-intrinsic" {
    fn x86_mm_max_ps(x: i32x4, y: i32x4) -> i32x4;
    //~^ ERROR intrinsic argument 1 has wrong type
    //~^^ ERROR intrinsic argument 2 has wrong type
    //~^^^ ERROR intrinsic return value has wrong type
}


fn main() {}
