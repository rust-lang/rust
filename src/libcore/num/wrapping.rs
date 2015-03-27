// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![allow(missing_docs)]

use ops::*;

use intrinsics::{overflowing_add, overflowing_sub, overflowing_mul};

use intrinsics::{i8_add_with_overflow, u8_add_with_overflow};
use intrinsics::{i16_add_with_overflow, u16_add_with_overflow};
use intrinsics::{i32_add_with_overflow, u32_add_with_overflow};
use intrinsics::{i64_add_with_overflow, u64_add_with_overflow};
use intrinsics::{i8_sub_with_overflow, u8_sub_with_overflow};
use intrinsics::{i16_sub_with_overflow, u16_sub_with_overflow};
use intrinsics::{i32_sub_with_overflow, u32_sub_with_overflow};
use intrinsics::{i64_sub_with_overflow, u64_sub_with_overflow};
use intrinsics::{i8_mul_with_overflow, u8_mul_with_overflow};
use intrinsics::{i16_mul_with_overflow, u16_mul_with_overflow};
use intrinsics::{i32_mul_with_overflow, u32_mul_with_overflow};
use intrinsics::{i64_mul_with_overflow, u64_mul_with_overflow};

pub trait WrappingOps {
    fn wrapping_add(self, rhs: Self) -> Self;
    fn wrapping_sub(self, rhs: Self) -> Self;
    fn wrapping_mul(self, rhs: Self) -> Self;
}

#[unstable(feature = "core", reason = "may be removed, renamed, or relocated")]
pub trait OverflowingOps {
    fn overflowing_add(self, rhs: Self) -> (Self, bool);
    fn overflowing_sub(self, rhs: Self) -> (Self, bool);
    fn overflowing_mul(self, rhs: Self) -> (Self, bool);
}

macro_rules! wrapping_impl {
    ($($t:ty)*) => ($(
        impl WrappingOps for $t {
            #[inline(always)]
            fn wrapping_add(self, rhs: $t) -> $t {
                unsafe {
                    overflowing_add(self, rhs)
                }
            }
            #[inline(always)]
            fn wrapping_sub(self, rhs: $t) -> $t {
                unsafe {
                    overflowing_sub(self, rhs)
                }
            }
            #[inline(always)]
            fn wrapping_mul(self, rhs: $t) -> $t {
                unsafe {
                    overflowing_mul(self, rhs)
                }
            }
        }
    )*)
}

wrapping_impl! { uint u8 u16 u32 u64 int i8 i16 i32 i64 }

#[unstable(feature = "core", reason = "may be removed, renamed, or relocated")]
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct Wrapping<T>(pub T);

impl<T:WrappingOps> Add for Wrapping<T> {
    type Output = Wrapping<T>;

    #[inline(always)]
    fn add(self, other: Wrapping<T>) -> Wrapping<T> {
        Wrapping(self.0.wrapping_add(other.0))
    }
}

impl<T:WrappingOps> Sub for Wrapping<T> {
    type Output = Wrapping<T>;

    #[inline(always)]
    fn sub(self, other: Wrapping<T>) -> Wrapping<T> {
        Wrapping(self.0.wrapping_sub(other.0))
    }
}

impl<T:WrappingOps> Mul for Wrapping<T> {
    type Output = Wrapping<T>;

    #[inline(always)]
    fn mul(self, other: Wrapping<T>) -> Wrapping<T> {
        Wrapping(self.0.wrapping_mul(other.0))
    }
}

impl<T:WrappingOps+Not<Output=T>> Not for Wrapping<T> {
    type Output = Wrapping<T>;

    fn not(self) -> Wrapping<T> {
        Wrapping(!self.0)
    }
}

impl<T:WrappingOps+BitXor<Output=T>> BitXor for Wrapping<T> {
    type Output = Wrapping<T>;

    #[inline(always)]
    fn bitxor(self, other: Wrapping<T>) -> Wrapping<T> {
        Wrapping(self.0 ^ other.0)
    }
}

impl<T:WrappingOps+BitOr<Output=T>> BitOr for Wrapping<T> {
    type Output = Wrapping<T>;

    #[inline(always)]
    fn bitor(self, other: Wrapping<T>) -> Wrapping<T> {
        Wrapping(self.0 | other.0)
    }
}

impl<T:WrappingOps+BitAnd<Output=T>> BitAnd for Wrapping<T> {
    type Output = Wrapping<T>;

    #[inline(always)]
    fn bitand(self, other: Wrapping<T>) -> Wrapping<T> {
        Wrapping(self.0 & other.0)
    }
}

impl<T:WrappingOps+Shl<uint,Output=T>> Shl<uint> for Wrapping<T> {
    type Output = Wrapping<T>;

    #[inline(always)]
    fn shl(self, other: uint) -> Wrapping<T> {
        Wrapping(self.0 << other)
    }
}

impl<T:WrappingOps+Shr<uint,Output=T>> Shr<uint> for Wrapping<T> {
    type Output = Wrapping<T>;

    #[inline(always)]
    fn shr(self, other: uint) -> Wrapping<T> {
        Wrapping(self.0 >> other)
    }
}

macro_rules! overflowing_impl {
    ($($t:ident)*) => ($(
        impl OverflowingOps for $t {
            #[inline(always)]
            fn overflowing_add(self, rhs: $t) -> ($t, bool) {
                unsafe {
                    concat_idents!($t, _add_with_overflow)(self, rhs)
                }
            }
            #[inline(always)]
            fn overflowing_sub(self, rhs: $t) -> ($t, bool) {
                unsafe {
                    concat_idents!($t, _sub_with_overflow)(self, rhs)
                }
            }
            #[inline(always)]
            fn overflowing_mul(self, rhs: $t) -> ($t, bool) {
                unsafe {
                    concat_idents!($t, _mul_with_overflow)(self, rhs)
                }
            }
        }
    )*)
}

overflowing_impl! { u8 u16 u32 u64 i8 i16 i32 i64 }

#[cfg(target_pointer_width = "64")]
impl OverflowingOps for usize {
    #[inline(always)]
    fn overflowing_add(self, rhs: usize) -> (usize, bool) {
        unsafe {
            let res = u64_add_with_overflow(self as u64, rhs as u64);
            (res.0 as usize, res.1)
        }
    }
    #[inline(always)]
    fn overflowing_sub(self, rhs: usize) -> (usize, bool) {
        unsafe {
            let res = u64_sub_with_overflow(self as u64, rhs as u64);
            (res.0 as usize, res.1)
        }
    }
    #[inline(always)]
    fn overflowing_mul(self, rhs: usize) -> (usize, bool) {
        unsafe {
            let res = u64_mul_with_overflow(self as u64, rhs as u64);
            (res.0 as usize, res.1)
        }
    }
}

#[cfg(target_pointer_width = "32")]
impl OverflowingOps for usize {
    #[inline(always)]
    fn overflowing_add(self, rhs: usize) -> (usize, bool) {
        unsafe {
            let res = u32_add_with_overflow(self as u32, rhs as u32);
            (res.0 as usize, res.1)
        }
    }
    #[inline(always)]
    fn overflowing_sub(self, rhs: usize) -> (usize, bool) {
        unsafe {
            let res = u32_sub_with_overflow(self as u32, rhs as u32);
            (res.0 as usize, res.1)
        }
    }
    #[inline(always)]
    fn overflowing_mul(self, rhs: usize) -> (usize, bool) {
        unsafe {
            let res = u32_mul_with_overflow(self as u32, rhs as u32);
            (res.0 as usize, res.1)
        }
    }
}

#[cfg(target_pointer_width = "64")]
impl OverflowingOps for isize {
    #[inline(always)]
    fn overflowing_add(self, rhs: isize) -> (isize, bool) {
        unsafe {
            let res = i64_add_with_overflow(self as i64, rhs as i64);
            (res.0 as isize, res.1)
        }
    }
    #[inline(always)]
    fn overflowing_sub(self, rhs: isize) -> (isize, bool) {
        unsafe {
            let res = i64_sub_with_overflow(self as i64, rhs as i64);
            (res.0 as isize, res.1)
        }
    }
    #[inline(always)]
    fn overflowing_mul(self, rhs: isize) -> (isize, bool) {
        unsafe {
            let res = i64_mul_with_overflow(self as i64, rhs as i64);
            (res.0 as isize, res.1)
        }
    }
}

#[cfg(target_pointer_width = "32")]
impl OverflowingOps for isize {
    #[inline(always)]
    fn overflowing_add(self, rhs: isize) -> (isize, bool) {
        unsafe {
            let res = i32_add_with_overflow(self as i32, rhs as i32);
            (res.0 as isize, res.1)
        }
    }
    #[inline(always)]
    fn overflowing_sub(self, rhs: isize) -> (isize, bool) {
        unsafe {
            let res = i32_sub_with_overflow(self as i32, rhs as i32);
            (res.0 as isize, res.1)
        }
    }
    #[inline(always)]
    fn overflowing_mul(self, rhs: isize) -> (isize, bool) {
        unsafe {
            let res = i32_mul_with_overflow(self as i32, rhs as i32);
            (res.0 as isize, res.1)
        }
    }
}
