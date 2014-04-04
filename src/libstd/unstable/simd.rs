// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! SIMD vectors

#![allow(non_camel_case_types)]

#[experimental]
#[simd]
pub struct i8x16(pub i8, pub i8, pub i8, pub i8,
                 pub i8, pub i8, pub i8, pub i8,
                 pub i8, pub i8, pub i8, pub i8,
                 pub i8, pub i8, pub i8, pub i8);

#[experimental]
#[simd]
pub struct i16x8(pub i16, pub i16, pub i16, pub i16,
                 pub i16, pub i16, pub i16, pub i16);

#[experimental]
#[simd]
pub struct i32x4(pub i32, pub i32, pub i32, pub i32);

#[experimental]
#[simd]
pub struct i64x2(pub i64, pub i64);

#[experimental]
#[simd]
pub struct u8x16(pub u8, pub u8, pub u8, pub u8,
                 pub u8, pub u8, pub u8, pub u8,
                 pub u8, pub u8, pub u8, pub u8,
                 pub u8, pub u8, pub u8, pub u8);

#[experimental]
#[simd]
pub struct u16x8(pub u16, pub u16, pub u16, pub u16,
                 pub u16, pub u16, pub u16, pub u16);

#[experimental]
#[simd]
pub struct u32x4(pub u32, pub u32, pub u32, pub u32);

#[experimental]
#[simd]
pub struct u64x2(pub u64, pub u64);

#[experimental]
#[simd]
pub struct f32x4(pub f32, pub f32, pub f32, pub f32);

#[experimental]
#[simd]
pub struct f64x2(pub f64, pub f64);
