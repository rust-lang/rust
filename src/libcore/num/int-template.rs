// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use T = self::inst::T;

use to_str::ToStr;
use from_str::FromStr;
use num::{ToStrRadix, FromStrRadix};
use num::strconv;
use num;
use prelude::*;

#[cfg(notest)] use cmp::{Eq, Ord};

pub use cmp::{min, max};

pub static bits : uint = inst::bits;
pub static bytes : uint = (inst::bits / 8);

pub static min_value: T = (-1 as T) << (bits - 1);
pub static max_value: T = min_value - 1 as T;

#[inline(always)]
pub fn add(x: T, y: T) -> T { x + y }
#[inline(always)]
pub fn sub(x: T, y: T) -> T { x - y }
#[inline(always)]
pub fn mul(x: T, y: T) -> T { x * y }
#[inline(always)]
pub fn div(x: T, y: T) -> T { x / y }

/**
 * Returns the remainder of y / x.
 *
 * # Examples
 * ~~~
 * assert!(int::rem(5 / 2) == 1);
 * ~~~
 *
 * When faced with negative numbers, the result copies the sign of the
 * dividend.
 *
 * ~~~
 * assert!(int::rem(2 / -3) ==  2);
 * ~~~
 *
 * ~~~
 * assert!(int::rem(-2 / 3) ==  -2);
 * ~~~
 *
 */
#[inline(always)]
pub fn rem(x: T, y: T) -> T { x % y }

#[inline(always)]
pub fn lt(x: T, y: T) -> bool { x < y }
#[inline(always)]
pub fn le(x: T, y: T) -> bool { x <= y }
#[inline(always)]
pub fn eq(x: T, y: T) -> bool { x == y }
#[inline(always)]
pub fn ne(x: T, y: T) -> bool { x != y }
#[inline(always)]
pub fn ge(x: T, y: T) -> bool { x >= y }
#[inline(always)]
pub fn gt(x: T, y: T) -> bool { x > y }

#[inline(always)]
pub fn is_positive(x: T) -> bool { x > 0 as T }
#[inline(always)]
pub fn is_negative(x: T) -> bool { x < 0 as T }
#[inline(always)]
pub fn is_nonpositive(x: T) -> bool { x <= 0 as T }
#[inline(always)]
pub fn is_nonnegative(x: T) -> bool { x >= 0 as T }

/**
 * Iterate over the range [`lo`..`hi`)
 *
 * # Arguments
 *
 * * `lo` - lower bound, inclusive
 * * `hi` - higher bound, exclusive
 *
 * # Examples
 * ~~~
 * let mut sum = 0;
 * for int::range(1, 5) |i| {
 *     sum += i;
 * }
 * assert!(sum == 10);
 * ~~~
 */
#[inline(always)]
/// Iterate over the range [`start`,`start`+`step`..`stop`)
pub fn range_step(start: T, stop: T, step: T, it: &fn(T) -> bool) {
    let mut i = start;
    if step == 0 {
        fail!(~"range_step called with step == 0");
    } else if step > 0 { // ascending
        while i < stop {
            if !it(i) { break }
            i += step;
        }
    } else { // descending
        while i > stop {
            if !it(i) { break }
            i += step;
        }
    }
}

#[inline(always)]
/// Iterate over the range [`lo`..`hi`)
pub fn range(lo: T, hi: T, it: &fn(T) -> bool) {
    range_step(lo, hi, 1 as T, it);
}

#[inline(always)]
/// Iterate over the range [`hi`..`lo`)
pub fn range_rev(hi: T, lo: T, it: &fn(T) -> bool) {
    range_step(hi, lo, -1 as T, it);
}

/// Computes the bitwise complement
#[inline(always)]
pub fn compl(i: T) -> T {
    -1 as T ^ i
}

/// Computes the absolute value
#[inline(always)]
pub fn abs(i: T) -> T {
    if is_negative(i) { -i } else { i }
}

#[cfg(notest)]
impl Ord for T {
    #[inline(always)]
    fn lt(&self, other: &T) -> bool { return (*self) < (*other); }
    #[inline(always)]
    fn le(&self, other: &T) -> bool { return (*self) <= (*other); }
    #[inline(always)]
    fn ge(&self, other: &T) -> bool { return (*self) >= (*other); }
    #[inline(always)]
    fn gt(&self, other: &T) -> bool { return (*self) > (*other); }
}

#[cfg(notest)]
impl Eq for T {
    #[inline(always)]
    fn eq(&self, other: &T) -> bool { return (*self) == (*other); }
    #[inline(always)]
    fn ne(&self, other: &T) -> bool { return (*self) != (*other); }
}

impl num::Zero for T {
    #[inline(always)]
    fn zero() -> T { 0 }
}

impl num::One for T {
    #[inline(always)]
    fn one() -> T { 1 }
}

#[cfg(notest)]
impl ops::Add<T,T> for T {
    fn add(&self, other: &T) -> T { *self + *other }
}
#[cfg(notest)]
impl ops::Sub<T,T> for T {
    fn sub(&self, other: &T) -> T { *self - *other }
}
#[cfg(notest)]
impl ops::Mul<T,T> for T {
    fn mul(&self, other: &T) -> T { *self * *other }
}
#[cfg(notest)]
impl ops::Div<T,T> for T {
    fn div(&self, other: &T) -> T { *self / *other }
}
#[cfg(notest)]
impl ops::Modulo<T,T> for T {
    fn modulo(&self, other: &T) -> T { *self % *other }
}
#[cfg(notest)]
impl ops::Neg<T> for T {
    fn neg(&self) -> T { -*self }
}

// String conversion functions and impl str -> num

/// Parse a string as a number in base 10.
#[inline(always)]
pub fn from_str(s: &str) -> Option<T> {
    strconv::from_str_common(s, 10u, true, false, false,
                         strconv::ExpNone, false, false)
}

/// Parse a string as a number in the given base.
#[inline(always)]
pub fn from_str_radix(s: &str, radix: uint) -> Option<T> {
    strconv::from_str_common(s, radix, true, false, false,
                         strconv::ExpNone, false, false)
}

/// Parse a byte slice as a number in the given base.
#[inline(always)]
pub fn parse_bytes(buf: &[u8], radix: uint) -> Option<T> {
    strconv::from_str_bytes_common(buf, radix, true, false, false,
                               strconv::ExpNone, false, false)
}

impl FromStr for T {
    #[inline(always)]
    fn from_str(s: &str) -> Option<T> {
        from_str(s)
    }
}

impl FromStrRadix for T {
    #[inline(always)]
    fn from_str_radix(s: &str, radix: uint) -> Option<T> {
        from_str_radix(s, radix)
    }
}

// String conversion functions and impl num -> str

/// Convert to a string as a byte slice in a given base.
#[inline(always)]
pub fn to_str_bytes<U>(n: T, radix: uint, f: &fn(v: &[u8]) -> U) -> U {
    let (buf, _) = strconv::to_str_bytes_common(&n, radix, false,
                            strconv::SignNeg, strconv::DigAll);
    f(buf)
}

/// Convert to a string in base 10.
#[inline(always)]
pub fn to_str(num: T) -> ~str {
    let (buf, _) = strconv::to_str_common(&num, 10u, false,
                                      strconv::SignNeg, strconv::DigAll);
    buf
}

/// Convert to a string in a given base.
#[inline(always)]
pub fn to_str_radix(num: T, radix: uint) -> ~str {
    let (buf, _) = strconv::to_str_common(&num, radix, false,
                                      strconv::SignNeg, strconv::DigAll);
    buf
}

impl ToStr for T {
    #[inline(always)]
    fn to_str(&self) -> ~str {
        to_str(*self)
    }
}

impl ToStrRadix for T {
    #[inline(always)]
    fn to_str_radix(&self, radix: uint) -> ~str {
        to_str_radix(*self, radix)
    }
}

#[test]
fn test_from_str() {
    assert!(from_str(~"0") == Some(0 as T));
    assert!(from_str(~"3") == Some(3 as T));
    assert!(from_str(~"10") == Some(10 as T));
    assert!(i32::from_str(~"123456789") == Some(123456789 as i32));
    assert!(from_str(~"00100") == Some(100 as T));

    assert!(from_str(~"-1") == Some(-1 as T));
    assert!(from_str(~"-3") == Some(-3 as T));
    assert!(from_str(~"-10") == Some(-10 as T));
    assert!(i32::from_str(~"-123456789") == Some(-123456789 as i32));
    assert!(from_str(~"-00100") == Some(-100 as T));

    assert!(from_str(~" ").is_none());
    assert!(from_str(~"x").is_none());
}

#[test]
fn test_parse_bytes() {
    use str::to_bytes;
    assert!(parse_bytes(to_bytes(~"123"), 10u) == Some(123 as T));
    assert!(parse_bytes(to_bytes(~"1001"), 2u) == Some(9 as T));
    assert!(parse_bytes(to_bytes(~"123"), 8u) == Some(83 as T));
    assert!(i32::parse_bytes(to_bytes(~"123"), 16u) == Some(291 as i32));
    assert!(i32::parse_bytes(to_bytes(~"ffff"), 16u) ==
                 Some(65535 as i32));
    assert!(i32::parse_bytes(to_bytes(~"FFFF"), 16u) ==
                 Some(65535 as i32));
    assert!(parse_bytes(to_bytes(~"z"), 36u) == Some(35 as T));
    assert!(parse_bytes(to_bytes(~"Z"), 36u) == Some(35 as T));

    assert!(parse_bytes(to_bytes(~"-123"), 10u) == Some(-123 as T));
    assert!(parse_bytes(to_bytes(~"-1001"), 2u) == Some(-9 as T));
    assert!(parse_bytes(to_bytes(~"-123"), 8u) == Some(-83 as T));
    assert!(i32::parse_bytes(to_bytes(~"-123"), 16u) ==
                 Some(-291 as i32));
    assert!(i32::parse_bytes(to_bytes(~"-ffff"), 16u) ==
                 Some(-65535 as i32));
    assert!(i32::parse_bytes(to_bytes(~"-FFFF"), 16u) ==
                 Some(-65535 as i32));
    assert!(parse_bytes(to_bytes(~"-z"), 36u) == Some(-35 as T));
    assert!(parse_bytes(to_bytes(~"-Z"), 36u) == Some(-35 as T));

    assert!(parse_bytes(to_bytes(~"Z"), 35u).is_none());
    assert!(parse_bytes(to_bytes(~"-9"), 2u).is_none());
}

#[test]
fn test_to_str() {
    assert!((to_str_radix(0 as T, 10u) == ~"0"));
    assert!((to_str_radix(1 as T, 10u) == ~"1"));
    assert!((to_str_radix(-1 as T, 10u) == ~"-1"));
    assert!((to_str_radix(127 as T, 16u) == ~"7f"));
    assert!((to_str_radix(100 as T, 10u) == ~"100"));

}

#[test]
fn test_int_to_str_overflow() {
    let mut i8_val: i8 = 127_i8;
    assert!((i8::to_str(i8_val) == ~"127"));

    i8_val += 1 as i8;
    assert!((i8::to_str(i8_val) == ~"-128"));

    let mut i16_val: i16 = 32_767_i16;
    assert!((i16::to_str(i16_val) == ~"32767"));

    i16_val += 1 as i16;
    assert!((i16::to_str(i16_val) == ~"-32768"));

    let mut i32_val: i32 = 2_147_483_647_i32;
    assert!((i32::to_str(i32_val) == ~"2147483647"));

    i32_val += 1 as i32;
    assert!((i32::to_str(i32_val) == ~"-2147483648"));

    let mut i64_val: i64 = 9_223_372_036_854_775_807_i64;
    assert!((i64::to_str(i64_val) == ~"9223372036854775807"));

    i64_val += 1 as i64;
    assert!((i64::to_str(i64_val) == ~"-9223372036854775808"));
}

#[test]
fn test_int_from_str_overflow() {
    let mut i8_val: i8 = 127_i8;
    assert!((i8::from_str(~"127") == Some(i8_val)));
    assert!((i8::from_str(~"128").is_none()));

    i8_val += 1 as i8;
    assert!((i8::from_str(~"-128") == Some(i8_val)));
    assert!((i8::from_str(~"-129").is_none()));

    let mut i16_val: i16 = 32_767_i16;
    assert!((i16::from_str(~"32767") == Some(i16_val)));
    assert!((i16::from_str(~"32768").is_none()));

    i16_val += 1 as i16;
    assert!((i16::from_str(~"-32768") == Some(i16_val)));
    assert!((i16::from_str(~"-32769").is_none()));

    let mut i32_val: i32 = 2_147_483_647_i32;
    assert!((i32::from_str(~"2147483647") == Some(i32_val)));
    assert!((i32::from_str(~"2147483648").is_none()));

    i32_val += 1 as i32;
    assert!((i32::from_str(~"-2147483648") == Some(i32_val)));
    assert!((i32::from_str(~"-2147483649").is_none()));

    let mut i64_val: i64 = 9_223_372_036_854_775_807_i64;
    assert!((i64::from_str(~"9223372036854775807") == Some(i64_val)));
    assert!((i64::from_str(~"9223372036854775808").is_none()));

    i64_val += 1 as i64;
    assert!((i64::from_str(~"-9223372036854775808") == Some(i64_val)));
    assert!((i64::from_str(~"-9223372036854775809").is_none()));
}

#[test]
pub fn test_num() {
    let ten: T = num::cast(10);
    let two: T = num::cast(2);

    assert!((ten.add(&two)    == num::cast(12)));
    assert!((ten.sub(&two)    == num::cast(8)));
    assert!((ten.mul(&two)    == num::cast(20)));
    assert!((ten.div(&two)    == num::cast(5)));
    assert!((ten.modulo(&two) == num::cast(0)));
}

#[test]
pub fn test_ranges() {
    let mut l = ~[];

    for range(0,3) |i| {
        l.push(i);
    }
    for range_rev(13,10) |i| {
        l.push(i);
    }
    for range_step(20,26,2) |i| {
        l.push(i);
    }
    for range_step(36,30,-2) |i| {
        l.push(i);
    }
    assert!(l == ~[0,1,2,
                        13,12,11,
                        20,22,24,
                        36,34,32]);

    // None of the `fail`s should execute.
    for range(10,0) |_i| {
        fail!(~"unreachable");
    }
    for range_rev(0,10) |_i| {
        fail!(~"unreachable");
    }
    for range_step(10,0,1) |_i| {
        fail!(~"unreachable");
    }
    for range_step(0,10,-1) |_i| {
        fail!(~"unreachable");
    }
}

#[test]
#[should_fail]
#[ignore(cfg(windows))]
fn test_range_step_zero_step() {
    for range_step(0,10,0) |_i| {}
}
