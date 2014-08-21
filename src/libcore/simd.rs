// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! SIMD vectors.
//!
//! These types can be used for accessing basic SIMD operations. Each of them
//! implements the standard arithmetic operator traits (Add, Sub, Mul, Div,
//! Rem, Shl, Shr) through compiler magic, rather than explicitly. Currently
//! comparison operators are not implemented. To use SSE3+, you must enable
//! the features, like `-C target-feature=sse3,sse4.1,sse4.2`, or a more
//! specific `target-cpu`. No other SIMD intrinsics or high-level wrappers are
//! provided beyond this module.
//!
//! ```rust
//! #[allow(experimental)];
//!
//! fn main() {
//!     use std::simd::f32x4;
//!     let a = f32x4(40.0, 41.0, 42.0, 43.0);
//!     let b = f32x4(1.0, 1.1, 3.4, 9.8);
//!     println!("{}", a + b);
//! }
//! ```
//!
//! ## Stability Note
//!
//! These are all experimental. The interface may change entirely, without
//! warning.

#![allow(non_camel_case_types)]
#![allow(missing_doc)]

#[experimental]
#[simd]
#[deriving(Show)]
#[repr(C)]
pub struct i8x16(pub i8, pub i8, pub i8, pub i8,
                 pub i8, pub i8, pub i8, pub i8,
                 pub i8, pub i8, pub i8, pub i8,
                 pub i8, pub i8, pub i8, pub i8);

#[experimental]
#[simd]
#[deriving(Show)]
#[repr(C)]
pub struct i16x8(pub i16, pub i16, pub i16, pub i16,
                 pub i16, pub i16, pub i16, pub i16);

#[experimental]
#[simd]
#[deriving(Show)]
#[repr(C)]
pub struct i32x4(pub i32, pub i32, pub i32, pub i32);

#[experimental]
#[simd]
#[deriving(Show)]
#[repr(C)]
pub struct i64x2(pub i64, pub i64);

#[experimental]
#[simd]
#[deriving(Show)]
#[repr(C)]
pub struct u8x16(pub u8, pub u8, pub u8, pub u8,
                 pub u8, pub u8, pub u8, pub u8,
                 pub u8, pub u8, pub u8, pub u8,
                 pub u8, pub u8, pub u8, pub u8);

#[experimental]
#[simd]
#[deriving(Show)]
#[repr(C)]
pub struct u16x8(pub u16, pub u16, pub u16, pub u16,
                 pub u16, pub u16, pub u16, pub u16);

#[experimental]
#[simd]
#[deriving(Show)]
#[repr(C)]
pub struct u32x4(pub u32, pub u32, pub u32, pub u32);

#[experimental]
#[simd]
#[deriving(Show)]
#[repr(C)]
pub struct u64x2(pub u64, pub u64);

#[experimental]
#[simd]
#[deriving(Show)]
#[repr(C)]
pub struct f32x4(pub f32, pub f32, pub f32, pub f32);

#[experimental]
#[simd]
#[deriving(Show)]
#[repr(C)]
pub struct f64x2(pub f64, pub f64);
