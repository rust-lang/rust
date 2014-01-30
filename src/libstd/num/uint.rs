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
use mem;
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

/// Returns the smallest power of 2 greater than or equal to `n`
#[inline]
pub fn next_power_of_two(n: uint) -> uint {
    let halfbits: uint = mem::size_of::<uint>() * 4u;
    let mut tmp: uint = n - 1u;
    let mut shift: uint = 1u;
    while shift <= halfbits { tmp |= tmp >> shift; shift <<= 1u; }
    tmp + 1u
}

/// Returns the smallest power of 2 greater than or equal to `n`
#[inline]
pub fn next_power_of_two_opt(n: uint) -> Option<uint> {
    let halfbits: uint = mem::size_of::<uint>() * 4u;
    let mut tmp: uint = n - 1u;
    let mut shift: uint = 1u;
    while shift <= halfbits { tmp |= tmp >> shift; shift <<= 1u; }
    tmp.checked_add(&1)
}

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
fn test_next_power_of_two() {
    assert!((next_power_of_two(0u) == 0u));
    assert!((next_power_of_two(1u) == 1u));
    assert!((next_power_of_two(2u) == 2u));
    assert!((next_power_of_two(3u) == 4u));
    assert!((next_power_of_two(4u) == 4u));
    assert!((next_power_of_two(5u) == 8u));
    assert!((next_power_of_two(6u) == 8u));
    assert!((next_power_of_two(7u) == 8u));
    assert!((next_power_of_two(8u) == 8u));
    assert!((next_power_of_two(9u) == 16u));
    assert!((next_power_of_two(10u) == 16u));
    assert!((next_power_of_two(11u) == 16u));
    assert!((next_power_of_two(12u) == 16u));
    assert!((next_power_of_two(13u) == 16u));
    assert!((next_power_of_two(14u) == 16u));
    assert!((next_power_of_two(15u) == 16u));
    assert!((next_power_of_two(16u) == 16u));
    assert!((next_power_of_two(17u) == 32u));
    assert!((next_power_of_two(18u) == 32u));
    assert!((next_power_of_two(19u) == 32u));
    assert!((next_power_of_two(20u) == 32u));
    assert!((next_power_of_two(21u) == 32u));
    assert!((next_power_of_two(22u) == 32u));
    assert!((next_power_of_two(23u) == 32u));
    assert!((next_power_of_two(24u) == 32u));
    assert!((next_power_of_two(25u) == 32u));
    assert!((next_power_of_two(26u) == 32u));
    assert!((next_power_of_two(27u) == 32u));
    assert!((next_power_of_two(28u) == 32u));
    assert!((next_power_of_two(29u) == 32u));
    assert!((next_power_of_two(30u) == 32u));
    assert!((next_power_of_two(31u) == 32u));
    assert!((next_power_of_two(32u) == 32u));
    assert!((next_power_of_two(33u) == 64u));
    assert!((next_power_of_two(34u) == 64u));
    assert!((next_power_of_two(35u) == 64u));
    assert!((next_power_of_two(36u) == 64u));
    assert!((next_power_of_two(37u) == 64u));
    assert!((next_power_of_two(38u) == 64u));
    assert!((next_power_of_two(39u) == 64u));
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
