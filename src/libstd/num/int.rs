// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for architecture-sized signed integers (`int` type)

#![allow(non_uppercase_statics)]

use prelude::*;

use default::Default;
use from_str::FromStr;
use num::{Bitwise, Bounded, CheckedAdd, CheckedSub, CheckedMul};
use num::{CheckedDiv, Zero, One, strconv};
use num::{ToStrRadix, FromStrRadix};
use option::{Option, Some, None};
use str;
use intrinsics;

#[cfg(target_word_size = "32")] int_module!(int, 32)
#[cfg(target_word_size = "64")] int_module!(int, 64)

#[cfg(target_word_size = "32")]
impl Bitwise for int {
    /// Returns the number of ones in the binary representation of the number.
    #[inline]
    fn count_ones(&self) -> int { (*self as i32).count_ones() as int }

    /// Returns the number of leading zeros in the in the binary representation
    /// of the number.
    #[inline]
    fn leading_zeros(&self) -> int { (*self as i32).leading_zeros() as int }

    /// Returns the number of trailing zeros in the in the binary representation
    /// of the number.
    #[inline]
    fn trailing_zeros(&self) -> int { (*self as i32).trailing_zeros() as int }
}

#[cfg(target_word_size = "64")]
impl Bitwise for int {
    /// Returns the number of ones in the binary representation of the number.
    #[inline]
    fn count_ones(&self) -> int { (*self as i64).count_ones() as int }

    /// Returns the number of leading zeros in the in the binary representation
    /// of the number.
    #[inline]
    fn leading_zeros(&self) -> int { (*self as i64).leading_zeros() as int }

    /// Returns the number of trailing zeros in the in the binary representation
    /// of the number.
    #[inline]
    fn trailing_zeros(&self) -> int { (*self as i64).trailing_zeros() as int }
}

#[cfg(target_word_size = "32")]
impl CheckedAdd for int {
    #[inline]
    fn checked_add(&self, v: &int) -> Option<int> {
        unsafe {
            let (x, y) = intrinsics::i32_add_with_overflow(*self as i32, *v as i32);
            if y { None } else { Some(x as int) }
        }
    }
}

#[cfg(target_word_size = "64")]
impl CheckedAdd for int {
    #[inline]
    fn checked_add(&self, v: &int) -> Option<int> {
        unsafe {
            let (x, y) = intrinsics::i64_add_with_overflow(*self as i64, *v as i64);
            if y { None } else { Some(x as int) }
        }
    }
}

#[cfg(target_word_size = "32")]
impl CheckedSub for int {
    #[inline]
    fn checked_sub(&self, v: &int) -> Option<int> {
        unsafe {
            let (x, y) = intrinsics::i32_sub_with_overflow(*self as i32, *v as i32);
            if y { None } else { Some(x as int) }
        }
    }
}

#[cfg(target_word_size = "64")]
impl CheckedSub for int {
    #[inline]
    fn checked_sub(&self, v: &int) -> Option<int> {
        unsafe {
            let (x, y) = intrinsics::i64_sub_with_overflow(*self as i64, *v as i64);
            if y { None } else { Some(x as int) }
        }
    }
}

#[cfg(target_word_size = "32")]
impl CheckedMul for int {
    #[inline]
    fn checked_mul(&self, v: &int) -> Option<int> {
        unsafe {
            let (x, y) = intrinsics::i32_mul_with_overflow(*self as i32, *v as i32);
            if y { None } else { Some(x as int) }
        }
    }
}

#[cfg(target_word_size = "64")]
impl CheckedMul for int {
    #[inline]
    fn checked_mul(&self, v: &int) -> Option<int> {
        unsafe {
            let (x, y) = intrinsics::i64_mul_with_overflow(*self as i64, *v as i64);
            if y { None } else { Some(x as int) }
        }
    }
}
