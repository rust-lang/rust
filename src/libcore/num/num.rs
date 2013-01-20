// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An interface for numeric types
use cmp::Eq;
use option::{None, Option, Some};

pub trait Num {
    // FIXME: Trait composition. (#2616)
    pure fn add(&self, other: &Self) -> Self;
    pure fn sub(&self, other: &Self) -> Self;
    pure fn mul(&self, other: &Self) -> Self;
    pure fn div(&self, other: &Self) -> Self;
    pure fn modulo(&self, other: &Self) -> Self;
    pure fn neg(&self) -> Self;

    pure fn to_int(&self) -> int;
    static pure fn from_int(n: int) -> Self;
}

pub trait IntConvertible {
    pure fn to_int(&self) -> int;
    static pure fn from_int(n: int) -> Self;
}

pub trait Zero {
    static pure fn zero() -> Self;
}

pub trait One {
    static pure fn one() -> Self;
}

pub trait Round {
    pure fn round(&self, mode: RoundMode) -> self;

    pure fn floor(&self) -> self;
    pure fn ceil(&self)  -> self;
    pure fn fract(&self) -> self;
}

pub enum RoundMode {
    RoundDown,
    RoundUp,
    RoundToZero,
    RoundFromZero
}

pub trait ToStrRadix {
    pub pure fn to_str_radix(&self, radix: uint) -> ~str;
}

pub trait FromStrRadix {
    static pub pure fn from_str_radix(str: &str, radix: uint) -> Option<self>;
}

// Generic math functions:

/// Dynamically calculates the value `inf` (`1/0`).
/// Can fail on integer types.
#[inline(always)]
pub pure fn infinity<T: Num One Zero>() -> T {
    let _0: T = Zero::zero();
    let _1: T = One::one();
    _1 / _0
}

/// Dynamically calculates the value `-inf` (`-1/0`).
/// Can fail on integer types.
#[inline(always)]
pub pure fn neg_infinity<T: Num One Zero>() -> T {
    let _0: T = Zero::zero();
    let _1: T = One::one();
    - _1 / _0
}

/// Dynamically calculates the value `NaN` (`0/0`).
/// Can fail on integer types.
#[inline(always)]
pub pure fn NaN<T: Num Zero>() -> T {
    let _0: T = Zero::zero();
    _0 / _0
}

/// Returns `true` if `num` has the value `inf` (`1/0`).
/// Can fail on integer types.
#[inline(always)]
pub pure fn is_infinity<T: Num One Zero Eq>(num: &T) -> bool {
    (*num) == (infinity::<T>())
}

/// Returns `true` if `num` has the value `-inf` (`-1/0`).
/// Can fail on integer types.
#[inline(always)]
pub pure fn is_neg_infinity<T: Num One Zero Eq>(num: &T) -> bool {
    (*num) == (neg_infinity::<T>())
}

/// Returns `true` if `num` has the value `NaN` (is not equal to itself).
#[inline(always)]
pub pure fn is_NaN<T: Num Eq>(num: &T) -> bool {
    (*num) != (*num)
}

/// Returns `true` if `num` has the value `-0` (`1/num == -1/0`).
/// Can fail on integer types.
#[inline(always)]
pub pure fn is_neg_zero<T: Num One Zero Eq>(num: &T) -> bool {
    let _1: T = One::one();
    let _0: T = Zero::zero();
    *num == _0 && is_neg_infinity(&(_1 / *num))
}

/**
 * Calculates a power to a given radix, optimized for uint `pow` and `radix`.
 *
 * Returns `radix^pow` as `T`.
 *
 * Note:
 * Also returns `1` for `0^0`, despite that technically being an
 * undefined number. The Reason for this is twofold:
 * - If code written to use this function cares about that special case, it's
 *   probably going to catch it before making the call.
 * - If code written to use this function doesn't care about it, it's
 *   probably assuming that `x^0` always equals `1`.
 */ 
pub pure fn pow_with_uint<T: Num One Zero>(radix: uint, pow: uint) -> T {
    let _0: T = Zero::zero();
    let _1: T = One::one();

    if pow   == 0u { return _1; }
    if radix == 0u { return _0; }
    let mut my_pow     = pow;
    let mut total      = _1;
    let mut multiplier = Num::from_int(radix as int);
    while (my_pow > 0u) {
        if my_pow % 2u == 1u {
            total *= multiplier;
        }
        my_pow     /= 2u;
        multiplier *= multiplier;
    }
    total
}