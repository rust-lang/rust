// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use T = self::inst::T;

use char;
use cmp::{Eq, Ord};
use cmp;
use from_str::FromStr;
use iter;
use num;
use num::Num::from_int;
use prelude::*;
use str;
use uint;
use vec;

pub const bits : uint = inst::bits;
pub const bytes : uint = (inst::bits / 8);

pub const min_value: T = (-1 as T) << (bits - 1);
pub const max_value: T = min_value - 1 as T;

#[inline(always)]
pub pure fn min(x: T, y: T) -> T { if x < y { x } else { y } }
#[inline(always)]
pub pure fn max(x: T, y: T) -> T { if x > y { x } else { y } }

#[inline(always)]
pub pure fn add(x: T, y: T) -> T { x + y }
#[inline(always)]
pub pure fn sub(x: T, y: T) -> T { x - y }
#[inline(always)]
pub pure fn mul(x: T, y: T) -> T { x * y }
#[inline(always)]
pub pure fn div(x: T, y: T) -> T { x / y }

/**
 * Returns the remainder of y / x.
 *
 * # Examples
 * ~~~
 * assert int::rem(5 / 2) == 1;
 * ~~~
 *
 * When faced with negative numbers, the result copies the sign of the
 * dividend.
 *
 * ~~~
 * assert int::rem(2 / -3) ==  2;
 * ~~~
 *
 * ~~~
 * assert int::rem(-2 / 3) ==  -2;
 * ~~~
 *
 */
#[inline(always)]
pub pure fn rem(x: T, y: T) -> T { x % y }

#[inline(always)]
pub pure fn lt(x: T, y: T) -> bool { x < y }
#[inline(always)]
pub pure fn le(x: T, y: T) -> bool { x <= y }
#[inline(always)]
pub pure fn eq(x: T, y: T) -> bool { x == y }
#[inline(always)]
pub pure fn ne(x: T, y: T) -> bool { x != y }
#[inline(always)]
pub pure fn ge(x: T, y: T) -> bool { x >= y }
#[inline(always)]
pub pure fn gt(x: T, y: T) -> bool { x > y }

#[inline(always)]
pub pure fn is_positive(x: T) -> bool { x > 0 as T }
#[inline(always)]
pub pure fn is_negative(x: T) -> bool { x < 0 as T }
#[inline(always)]
pub pure fn is_nonpositive(x: T) -> bool { x <= 0 as T }
#[inline(always)]
pub pure fn is_nonnegative(x: T) -> bool { x >= 0 as T }

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
 * assert sum == 10;
 * ~~~
 */
#[inline(always)]
pub pure fn range(lo: T, hi: T, it: fn(T) -> bool) {
    let mut i = lo;
    while i < hi {
        if !it(i) { break }
        i += 1 as T;
    }
}

/// Computes the bitwise complement
#[inline(always)]
pub pure fn compl(i: T) -> T {
    -1 as T ^ i
}

/// Computes the absolute value
#[inline(always)]
pub pure fn abs(i: T) -> T {
    if is_negative(i) { -i } else { i }
}

#[cfg(notest)]
impl T : Ord {
    #[inline(always)]
    pure fn lt(&self, other: &T) -> bool { return (*self) < (*other); }
    #[inline(always)]
    pure fn le(&self, other: &T) -> bool { return (*self) <= (*other); }
    #[inline(always)]
    pure fn ge(&self, other: &T) -> bool { return (*self) >= (*other); }
    #[inline(always)]
    pure fn gt(&self, other: &T) -> bool { return (*self) > (*other); }
}

#[cfg(notest)]
impl T : Eq {
    #[inline(always)]
    pure fn eq(&self, other: &T) -> bool { return (*self) == (*other); }
    #[inline(always)]
    pure fn ne(&self, other: &T) -> bool { return (*self) != (*other); }
}

impl T: num::Num {
    #[inline(always)]
    pure fn add(&self, other: &T)    -> T { return *self + *other; }
    #[inline(always)]
    pure fn sub(&self, other: &T)    -> T { return *self - *other; }
    #[inline(always)]
    pure fn mul(&self, other: &T)    -> T { return *self * *other; }
    #[inline(always)]
    pure fn div(&self, other: &T)    -> T { return *self / *other; }
    #[inline(always)]
    pure fn modulo(&self, other: &T) -> T { return *self % *other; }
    #[inline(always)]
    pure fn neg(&self)              -> T { return -*self;        }

    #[inline(always)]
    pure fn to_int(&self)         -> int { return *self as int; }
    #[inline(always)]
    static pure fn from_int(n: int) -> T   { return n as T;      }
}

impl T: num::Zero {
    #[inline(always)]
    static pure fn zero() -> T { 0 }
}

impl T: num::One {
    #[inline(always)]
    static pure fn one() -> T { 1 }
}

impl T: iter::Times {
    #[inline(always)]
    #[doc = "A convenience form for basic iteration. Given a variable `x` \
        of any numeric type, the expression `for x.times { /* anything */ }` \
        will execute the given function exactly x times. If we assume that \
        `x` is an int, this is functionally equivalent to \
        `for int::range(0, x) |_i| { /* anything */ }`."]
    pure fn times(&self, it: fn() -> bool) {
        if is_negative(*self) {
            fail fmt!("The .times method expects a nonnegative number, \
                       but found %?", self);
        }
        let mut i = *self;
        while i > 0 {
            if !it() { break }
            i -= 1;
        }
    }
}

/**
 * Parse a buffer of bytes
 *
 * # Arguments
 *
 * * buf - A byte buffer
 * * radix - The base of the number
 */
pub pure fn parse_bytes(buf: &[u8], radix: uint) -> Option<T> {
    if vec::len(buf) == 0u { return None; }
    let mut i = vec::len(buf) - 1u;
    let mut start = 0u;
    let mut power = 1 as T;

    if buf[0] == ('-' as u8) {
        power = -1 as T;
        start = 1u;
    }
    let mut n = 0 as T;
    loop {
        match char::to_digit(buf[i] as char, radix) {
          Some(d) => n += (d as T) * power,
          None => return None
        }
        power *= radix as T;
        if i <= start { return Some(n); }
        i -= 1u;
    };
}

/// Parse a string to an int
#[inline(always)]
pub pure fn from_str(s: &str) -> Option<T>
{
    parse_bytes(str::to_bytes(s), 10u)
}

impl T : FromStr {
    #[inline(always)]
    static pure fn from_str(s: &str) -> Option<T> { from_str(s) }
}

/// Convert to a string in a given base
#[inline(always)]
pub pure fn to_str(n: T, radix: uint) -> ~str {
    do to_str_bytes(n, radix) |slice| {
        do vec::as_imm_buf(slice) |p, len| {
            unsafe { str::raw::from_buf_len(p, len) }
        }
    }
}

#[inline(always)]
pub pure fn to_str_bytes<U>(n: T, radix: uint, f: fn(v: &[u8]) -> U) -> U {
    if n < 0 as T {
        uint::to_str_bytes(true, -n as uint, radix, f)
    } else {
        uint::to_str_bytes(false, n as uint, radix, f)
    }
}

/// Convert to a string
#[inline(always)]
pub pure fn str(i: T) -> ~str { return to_str(i, 10u); }

#[test]
fn test_from_str() {
    assert from_str(~"0") == Some(0 as T);
    assert from_str(~"3") == Some(3 as T);
    assert from_str(~"10") == Some(10 as T);
    assert from_str(~"123456789") == Some(123456789 as T);
    assert from_str(~"00100") == Some(100 as T);

    assert from_str(~"-1") == Some(-1 as T);
    assert from_str(~"-3") == Some(-3 as T);
    assert from_str(~"-10") == Some(-10 as T);
    assert from_str(~"-123456789") == Some(-123456789 as T);
    assert from_str(~"-00100") == Some(-100 as T);

    assert from_str(~" ").is_none();
    assert from_str(~"x").is_none();
}

#[test]
fn test_parse_bytes() {
    use str::to_bytes;
    assert parse_bytes(to_bytes(~"123"), 10u) == Some(123 as T);
    assert parse_bytes(to_bytes(~"1001"), 2u) == Some(9 as T);
    assert parse_bytes(to_bytes(~"123"), 8u) == Some(83 as T);
    assert parse_bytes(to_bytes(~"123"), 16u) == Some(291 as T);
    assert parse_bytes(to_bytes(~"ffff"), 16u) == Some(65535 as T);
    assert parse_bytes(to_bytes(~"FFFF"), 16u) == Some(65535 as T);
    assert parse_bytes(to_bytes(~"z"), 36u) == Some(35 as T);
    assert parse_bytes(to_bytes(~"Z"), 36u) == Some(35 as T);

    assert parse_bytes(to_bytes(~"-123"), 10u) == Some(-123 as T);
    assert parse_bytes(to_bytes(~"-1001"), 2u) == Some(-9 as T);
    assert parse_bytes(to_bytes(~"-123"), 8u) == Some(-83 as T);
    assert parse_bytes(to_bytes(~"-123"), 16u) == Some(-291 as T);
    assert parse_bytes(to_bytes(~"-ffff"), 16u) == Some(-65535 as T);
    assert parse_bytes(to_bytes(~"-FFFF"), 16u) == Some(-65535 as T);
    assert parse_bytes(to_bytes(~"-z"), 36u) == Some(-35 as T);
    assert parse_bytes(to_bytes(~"-Z"), 36u) == Some(-35 as T);

    assert parse_bytes(to_bytes(~"Z"), 35u).is_none();
    assert parse_bytes(to_bytes(~"-9"), 2u).is_none();
}

#[test]
fn test_to_str() {
    assert (to_str(0 as T, 10u) == ~"0");
    assert (to_str(1 as T, 10u) == ~"1");
    assert (to_str(-1 as T, 10u) == ~"-1");
    assert (to_str(127 as T, 16u) == ~"7f");
    assert (to_str(100 as T, 10u) == ~"100");
}

#[test]
fn test_interfaces() {
    fn test<U:num::Num cmp::Eq>(ten: U) {
        assert (ten.to_int() == 10);

        let two: U = from_int(2);
        assert (two.to_int() == 2);

        assert (ten.add(&two) == from_int(12));
        assert (ten.sub(&two) == from_int(8));
        assert (ten.mul(&two) == from_int(20));
        assert (ten.div(&two) == from_int(5));
        assert (ten.modulo(&two) == from_int(0));
        assert (ten.neg() == from_int(-10));
    }

    test(10 as T);
}

#[test]
fn test_times() {
    use iter::Times;
    let ten = 10 as T;
    let mut accum = 0;
    for ten.times { accum += 1; }
    assert (accum == 10);
}

#[test]
#[should_fail]
#[ignore(cfg(windows))]
fn test_times_negative() {
    use iter::Times;
    for (-10).times { log(error, ~"nope!"); }
}
