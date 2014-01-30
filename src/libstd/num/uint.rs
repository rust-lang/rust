// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for architecture-sized unsigned integers (`uint` type)

#[allow(non_uppercase_statics)];

use prelude::*;

use default::Default;
use num::{Bitwise, Bounded};
use num::{CheckedAdd, CheckedSub, CheckedMul};
use num::{CheckedDiv, Zero, One, strconv};
use num::{ToStrRadix, FromStrRadix};
use option::{Option, Some, None};
use str;
use unstable::intrinsics;

uint_module!(uint, int, ::int::BITS)

///
/// Divide two numbers, return the result, rounded up.
///
/// # Arguments
///
/// * x - an integer
/// * y - an integer distinct from 0u
///
/// # Return value
///
/// The smallest integer `q` such that `x/y <= q`.
///
pub fn div_ceil(x: uint, y: uint) -> uint {
    let div = x / y;
    if x % y == 0u { div }
    else { div + 1u }
}

///
/// Divide two numbers, return the result, rounded to the closest integer.
///
/// # Arguments
///
/// * x - an integer
/// * y - an integer distinct from 0u
///
/// # Return value
///
/// The integer `q` closest to `x/y`.
///
pub fn div_round(x: uint, y: uint) -> uint {
    let div = x / y;
    if x % y * 2u  < y { div }
    else { div + 1u }
}

///
/// Divide two numbers, return the result, rounded down.
///
/// Note: This is the same function as `div`.
///
/// # Arguments
///
/// * x - an integer
/// * y - an integer distinct from 0u
///
/// # Return value
///
/// The smallest integer `q` such that `x/y <= q`. This
/// is either `x/y` or `x/y + 1`.
///
pub fn div_floor(x: uint, y: uint) -> uint { return x / y; }

#[cfg(target_word_size = "32")]
impl CheckedAdd for uint {
    #[inline]
    fn checked_add(&self, v: &uint) -> Option<uint> {
        unsafe {
            let (x, y) = intrinsics::u32_add_with_overflow(*self as u32, *v as u32);
            if y { None } else { Some(x as uint) }
        }
    }
}

#[cfg(target_word_size = "64")]
impl CheckedAdd for uint {
    #[inline]
    fn checked_add(&self, v: &uint) -> Option<uint> {
        unsafe {
            let (x, y) = intrinsics::u64_add_with_overflow(*self as u64, *v as u64);
            if y { None } else { Some(x as uint) }
        }
    }
}

#[cfg(target_word_size = "32")]
impl CheckedSub for uint {
    #[inline]
    fn checked_sub(&self, v: &uint) -> Option<uint> {
        unsafe {
            let (x, y) = intrinsics::u32_sub_with_overflow(*self as u32, *v as u32);
            if y { None } else { Some(x as uint) }
        }
    }
}

#[cfg(target_word_size = "64")]
impl CheckedSub for uint {
    #[inline]
    fn checked_sub(&self, v: &uint) -> Option<uint> {
        unsafe {
            let (x, y) = intrinsics::u64_sub_with_overflow(*self as u64, *v as u64);
            if y { None } else { Some(x as uint) }
        }
    }
}

#[cfg(target_word_size = "32")]
impl CheckedMul for uint {
    #[inline]
    fn checked_mul(&self, v: &uint) -> Option<uint> {
        unsafe {
            let (x, y) = intrinsics::u32_mul_with_overflow(*self as u32, *v as u32);
            if y { None } else { Some(x as uint) }
        }
    }
}

#[cfg(target_word_size = "64")]
impl CheckedMul for uint {
    #[inline]
    fn checked_mul(&self, v: &uint) -> Option<uint> {
        unsafe {
            let (x, y) = intrinsics::u64_mul_with_overflow(*self as u64, *v as u64);
            if y { None } else { Some(x as uint) }
        }
    }
}

#[test]
fn test_overflows() {
    use uint;
    assert!((uint::MAX > 0u));
    assert!((uint::MIN <= 0u));
    assert!((uint::MIN + uint::MAX + 1u == 0u));
}

#[test]
fn test_div() {
    assert!((div_floor(3u, 4u) == 0u));
    assert!((div_ceil(3u, 4u)  == 1u));
    assert!((div_round(3u, 4u) == 1u));
}
