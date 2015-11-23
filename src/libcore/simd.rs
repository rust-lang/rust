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
//! These types can be used for accessing basic SIMD operations. Currently
//! comparison operators are not implemented. To use SSE3+, you must enable
//! the features, like `-C target-feature=sse3,sse4.1,sse4.2`, or a more
//! specific `target-cpu`. No other SIMD intrinsics or high-level wrappers are
//! provided beyond this module.
//!
//! # Stability Note
//!
//! These are all experimental. The interface may change entirely, without
//! warning.

#![unstable(feature = "core_simd",
            reason = "needs an RFC to flesh out the design",
            issue = "27731")]
#![rustc_deprecated(since = "1.3.0",
              reason = "use the external `simd` crate instead")]

#![allow(non_camel_case_types)]
#![allow(missing_docs)]
#![allow(deprecated)]

use ops::{Add, Sub, Mul, Div, Shl, Shr, BitAnd, BitOr, BitXor};

// FIXME(stage0): the contents of macro can be inlined.
// ABIs are verified as valid as soon as they are parsed, i.e. before
// `cfg` stripping. The `platform-intrinsic` ABI is new, so stage0
// doesn't know about it, but it still errors out when it hits it
// (despite this being in a `cfg(not(stage0))` module).
macro_rules! argh {
    () => {
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
    }
}
argh!();

#[repr(simd)]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct i8x16(pub i8, pub i8, pub i8, pub i8,
                 pub i8, pub i8, pub i8, pub i8,
                 pub i8, pub i8, pub i8, pub i8,
                 pub i8, pub i8, pub i8, pub i8);

#[repr(simd)]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct i16x8(pub i16, pub i16, pub i16, pub i16,
                 pub i16, pub i16, pub i16, pub i16);

#[repr(simd)]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct i32x4(pub i32, pub i32, pub i32, pub i32);

#[repr(simd)]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct i64x2(pub i64, pub i64);

#[repr(simd)]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct u8x16(pub u8, pub u8, pub u8, pub u8,
                 pub u8, pub u8, pub u8, pub u8,
                 pub u8, pub u8, pub u8, pub u8,
                 pub u8, pub u8, pub u8, pub u8);

#[repr(simd)]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct u16x8(pub u16, pub u16, pub u16, pub u16,
                 pub u16, pub u16, pub u16, pub u16);

#[repr(simd)]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct u32x4(pub u32, pub u32, pub u32, pub u32);

#[repr(simd)]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct u64x2(pub u64, pub u64);

#[repr(simd)]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct f32x4(pub f32, pub f32, pub f32, pub f32);

#[repr(simd)]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct f64x2(pub f64, pub f64);

macro_rules! impl_traits {
    ($($trayt: ident, $method: ident, $func: ident: $($ty: ty),*;)*) => {
        $($(
            impl $trayt<$ty> for $ty {
                type Output = Self;
                fn $method(self, other: Self) -> Self {
                    unsafe {
                        $func(self, other)
                    }
                }
            }
            )*)*
    }
}

impl_traits! {
    Add, add, simd_add: u8x16, u16x8, u32x4, u64x2, i8x16, i16x8, i32x4, i64x2, f32x4, f64x2;
    Sub, sub, simd_sub: u8x16, u16x8, u32x4, u64x2, i8x16, i16x8, i32x4, i64x2, f32x4, f64x2;
    Mul, mul, simd_mul: u8x16, u16x8, u32x4, u64x2, i8x16, i16x8, i32x4, i64x2, f32x4, f64x2;

    Div, div, simd_div: f32x4, f64x2;

    Shl, shl, simd_shl: u8x16, u16x8, u32x4, u64x2, i8x16, i16x8, i32x4, i64x2;
    Shr, shr, simd_shr: u8x16, u16x8, u32x4, u64x2, i8x16, i16x8, i32x4, i64x2;
    BitAnd, bitand, simd_and: u8x16, u16x8, u32x4, u64x2, i8x16, i16x8, i32x4, i64x2;
    BitOr, bitor, simd_or: u8x16, u16x8, u32x4, u64x2, i8x16, i16x8, i32x4, i64x2;
    BitXor, bitxor, simd_xor: u8x16, u16x8, u32x4, u64x2, i8x16, i16x8, i32x4, i64x2;
}
