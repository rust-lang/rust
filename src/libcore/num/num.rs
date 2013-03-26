// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An interface for numeric types
use cmp::Ord;
use ops::{Div, Mul, Neg};
use option::Option;
use kinds::Copy;

pub mod strconv;

pub trait IntConvertible {
    fn to_int(&self) -> int;
    fn from_int(n: int) -> Self;
}

pub trait Zero {
    fn zero() -> Self;
}

pub trait One {
    fn one() -> Self;
}

pub fn abs<T:Ord + Zero + Neg<T>>(v: T) -> T {
    if v < Zero::zero() { v.neg() } else { v }
}

pub trait Round {
    fn round(&self, mode: RoundMode) -> Self;

    fn floor(&self) -> Self;
    fn ceil(&self)  -> Self;
    fn fract(&self) -> Self;
}

pub enum RoundMode {
    RoundDown,
    RoundUp,
    RoundToZero,
    RoundFromZero
}

/**
 * Cast a number the the enclosing type
 *
 * # Example
 *
 * ~~~
 * let twenty: f32 = num::cast(0x14);
 * fail_unless!(twenty == 20f32);
 * ~~~
 */
#[inline(always)]
pub fn cast<T:NumCast,U:NumCast>(n: T) -> U {
    NumCast::from(n)
}

/**
 * An interface for generic numeric type casts
 */
pub trait NumCast {
    fn from<T:NumCast>(n: T) -> Self;

    fn to_u8(&self) -> u8;
    fn to_u16(&self) -> u16;
    fn to_u32(&self) -> u32;
    fn to_u64(&self) -> u64;
    fn to_uint(&self) -> uint;

    fn to_i8(&self) -> i8;
    fn to_i16(&self) -> i16;
    fn to_i32(&self) -> i32;
    fn to_i64(&self) -> i64;
    fn to_int(&self) -> int;

    fn to_f32(&self) -> f32;
    fn to_f64(&self) -> f64;
    fn to_float(&self) -> float;
}

pub trait ToStrRadix {
    pub fn to_str_radix(&self, radix: uint) -> ~str;
}

pub trait FromStrRadix {
    pub fn from_str_radix(str: &str, radix: uint) -> Option<Self>;
}

// Generic math functions:

/**
 * Calculates a power to a given radix, optimized for uint `pow` and `radix`.
 *
 * Returns `radix^pow` as `T`.
 *
 * Note:
 * Also returns `1` for `0^0`, despite that technically being an
 * undefined number. The reason for this is twofold:
 * - If code written to use this function cares about that special case, it's
 *   probably going to catch it before making the call.
 * - If code written to use this function doesn't care about it, it's
 *   probably assuming that `x^0` always equals `1`.
 */
pub fn pow_with_uint<T:NumCast+One+Zero+Copy+Div<T,T>+Mul<T,T>>(
    radix: uint, pow: uint) -> T {
    let _0: T = Zero::zero();
    let _1: T = One::one();

    if pow   == 0u { return _1; }
    if radix == 0u { return _0; }
    let mut my_pow     = pow;
    let mut total      = _1;
    let mut multiplier = cast(radix as int);
    while (my_pow > 0u) {
        if my_pow % 2u == 1u {
            total *= multiplier;
        }
        my_pow     /= 2u;
        multiplier *= multiplier;
    }
    total
}

