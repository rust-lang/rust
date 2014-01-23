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

#[allow(non_camel_case_types)];

#[experimental]
#[simd]
pub struct i8x16(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8);

#[experimental]
#[simd]
pub struct i16x8(i16, i16, i16, i16, i16, i16, i16, i16);

#[experimental]
#[simd]
pub struct i32x4(i32, i32, i32, i32);

#[experimental]
#[simd]
pub struct i64x2(i64, i64);

#[experimental]
#[simd]
pub struct u8x16(u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8);

#[experimental]
#[simd]
pub struct u16x8(u16, u16, u16, u16, u16, u16, u16, u16);

#[experimental]
#[simd]
pub struct u32x4(u32, u32, u32, u32);

#[experimental]
#[simd]
pub struct u64x2(u64, u64);

#[experimental]
#[simd]
pub struct f32x4(f32, f32, f32, f32);

#[experimental]
#[simd]
pub struct f64x2(f64, f64);
