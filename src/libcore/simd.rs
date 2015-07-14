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
//! #![feature(core_simd)]
//!
//! fn main() {
//!     use std::simd::f32x4;
//!     let a = f32x4(40.0, 41.0, 42.0, 43.0);
//!     let b = f32x4(1.0, 1.1, 3.4, 9.8);
//!     println!("{:?}", a + b);
//! }
//! ```
//!
//! # Stability Note
//!
//! These are all experimental. The interface may change entirely, without
//! warning.

#![unstable(feature = "core_simd",
            reason = "needs an RFC to flesh out the design")]

#![allow(non_camel_case_types)]
#![allow(missing_docs)]

#[cfg(all(target_os = "nacl", target_arch = "le32"))]
use {cmp, ops};

macro_rules! pnacl_abi_workaround_arithmetic (
    ($ty:ident) => {
        #[cfg(all(target_os = "nacl", target_arch = "le32"))]
        impl cmp::PartialEq for $ty {
            fn eq(&self, rhs: &$ty) -> bool {
                self.0 == rhs.0 && self.1 == rhs.1
            }
        }
        #[cfg(all(target_os = "nacl", target_arch = "le32"))]
        impl ops::Add for $ty {
            type Output = $ty;
            fn add(self, rhs: $ty) -> $ty {
                $ty(self.0 + rhs.0, self.1 + rhs.1)
            }
        }
        #[cfg(all(target_os = "nacl", target_arch = "le32"))]
        impl ops::Sub for $ty {
            type Output = $ty;
            fn sub(self, rhs: $ty) -> $ty {
                $ty(self.0 - rhs.0, self.1 - rhs.1)
            }
        }
        #[cfg(all(target_os = "nacl", target_arch = "le32"))]
        impl ops::Mul for $ty {
            type Output = $ty;
            fn mul(self, rhs: $ty) -> $ty {
                $ty(self.0 * rhs.0, self.1 * rhs.1)
            }
        }
        #[cfg(all(target_os = "nacl", target_arch = "le32"))]
        impl ops::Div for $ty {
            type Output = $ty;
            fn div(self, rhs: $ty) -> $ty {
                $ty(self.0 / rhs.0, self.1 / rhs.1)
            }
        }
        #[cfg(all(target_os = "nacl", target_arch = "le32"))]
        impl ops::Rem for $ty {
            type Output = $ty;
            fn rem(self, rhs: $ty) -> $ty {
                $ty(self.0 % rhs.0, self.1 % rhs.1)
            }
        }
    }
);
macro_rules! pnacl_abi_workaround_bit (
    ($ty:ident) => {
        #[cfg(all(target_os = "nacl", target_arch = "le32"))]
        impl ops::BitAnd for $ty {
            type Output = $ty;
            fn bitand(self, rhs: $ty) -> $ty {
                $ty(self.0 & rhs.0, self.1 & rhs.1)
            }
        }
        #[cfg(all(target_os = "nacl", target_arch = "le32"))]
        impl ops::BitOr for $ty {
            type Output = $ty;
            fn bitor(self, rhs: $ty) -> $ty {
                $ty(self.0 | rhs.0, self.1 | rhs.1)
            }
        }
        #[cfg(all(target_os = "nacl", target_arch = "le32"))]
        impl ops::BitXor for $ty {
            type Output = $ty;
            fn bitxor(self, rhs: $ty) -> $ty {
                $ty(self.0 ^ rhs.0, self.1 ^ rhs.1)
            }
        }
        #[cfg(all(target_os = "nacl", target_arch = "le32"))]
        impl ops::Shl<$ty> for $ty {
            type Output = $ty;
            fn shl(self, rhs: $ty) -> $ty {
                $ty(self.0 << rhs.0, self.1 << rhs.1)
            }
        }
        #[cfg(all(target_os = "nacl", target_arch = "le32"))]
        impl ops::Shr<$ty> for $ty {
            type Output = $ty;
            fn shr(self, rhs: $ty) -> $ty {
                $ty(self.0 >> rhs.0, self.1 >> rhs.1)
            }
        }
        #[cfg(all(target_os = "nacl", target_arch = "le32"))]
        impl ops::Not for $ty {
            type Output = $ty;
            fn not(self) -> $ty {
                $ty(!self.0, !self.1)
            }
        }


        #[cfg(all(target_os = "nacl", target_arch = "le32"))]
        impl cmp::Eq for $ty { }
    }
);
macro_rules! pnacl_abi_workaround_signed (
    ($ty:ident) => {
        #[cfg(all(target_os = "nacl", target_arch = "le32"))]
        impl ops::Neg for $ty {
            type Output = $ty;
            fn neg(self) -> $ty {
                $ty(-self.0, -self.1)
            }
        }
    }
);

#[simd]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct i8x16(pub i8, pub i8, pub i8, pub i8,
                 pub i8, pub i8, pub i8, pub i8,
                 pub i8, pub i8, pub i8, pub i8,
                 pub i8, pub i8, pub i8, pub i8);

#[simd]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct i16x8(pub i16, pub i16, pub i16, pub i16,
                 pub i16, pub i16, pub i16, pub i16);

#[simd]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct i32x4(pub i32, pub i32, pub i32, pub i32);

// The PNaCl ABI doesn't currently allow this type.
#[cfg_attr(not(all(target_os = "nacl", target_arch = "le32")), simd)]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct i64x2(pub i64, pub i64);
pnacl_abi_workaround_arithmetic!(i64x2);
pnacl_abi_workaround_bit!(i64x2);
pnacl_abi_workaround_signed!(i64x2);

#[simd]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct u8x16(pub u8, pub u8, pub u8, pub u8,
                 pub u8, pub u8, pub u8, pub u8,
                 pub u8, pub u8, pub u8, pub u8,
                 pub u8, pub u8, pub u8, pub u8);

#[simd]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct u16x8(pub u16, pub u16, pub u16, pub u16,
                 pub u16, pub u16, pub u16, pub u16);

#[simd]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct u32x4(pub u32, pub u32, pub u32, pub u32);

// The PNaCl ABI doesn't currently allow this type.
#[cfg_attr(not(all(target_os = "nacl", target_arch = "le32")), simd)]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct u64x2(pub u64, pub u64);
pnacl_abi_workaround_arithmetic!(u64x2);
pnacl_abi_workaround_bit!(u64x2);

#[simd]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct f32x4(pub f32, pub f32, pub f32, pub f32);

// The PNaCl ABI doesn't currently allow this type.
#[cfg_attr(not(all(target_os = "nacl", target_arch = "le32")), simd)]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct f64x2(pub f64, pub f64);
pnacl_abi_workaround_arithmetic!(f64x2);
pnacl_abi_workaround_signed!(f64x2);
