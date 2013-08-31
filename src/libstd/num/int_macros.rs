// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME(#4375): this shouldn't have to be a nested module named 'generated'

#[macro_escape];

macro_rules! int_module (($T:ty, $bits:expr) => (mod generated {

#[allow(non_uppercase_statics)];

use num::{ToStrRadix, FromStrRadix};
use num::{CheckedDiv, Zero, One, strconv};
use prelude::*;
use str;

pub use cmp::{min, max};

pub static bits : uint = $bits;
pub static bytes : uint = ($bits / 8);

pub static min_value: $T = (-1 as $T) << (bits - 1);
pub static max_value: $T = min_value - 1 as $T;

impl CheckedDiv for $T {
    #[inline]
    fn checked_div(&self, v: &$T) -> Option<$T> {
        if *v == 0 || (*self == min_value && *v == -1) {
            None
        } else {
            Some(self / *v)
        }
    }
}

enum Range { Closed, HalfOpen }

#[inline]
///
/// Iterate through a range with a given step value.
///
/// Let `term` denote the closed interval `[stop-step,stop]` if `r` is Closed;
/// otherwise `term` denotes the half-open interval `[stop-step,stop)`.
/// Iterates through the range `[x_0, x_1, ..., x_n]` where
/// `x_j == start + step*j`, and `x_n` lies in the interval `term`.
///
/// If no such nonnegative integer `n` exists, then the iteration range
/// is empty.
///
fn range_step_core(start: $T, stop: $T, step: $T, r: Range, it: &fn($T) -> bool) -> bool {
    let mut i = start;
    if step == 0 {
        fail!(~"range_step called with step == 0");
    } else if step == (1 as $T) { // elide bounds check to tighten loop
        while i < stop {
            if !it(i) { return false; }
            // no need for overflow check;
            // cannot have i + 1 > max_value because i < stop <= max_value
            i += (1 as $T);
        }
    } else if step == (-1 as $T) { // elide bounds check to tighten loop
        while i > stop {
            if !it(i) { return false; }
            // no need for underflow check;
            // cannot have i - 1 < min_value because i > stop >= min_value
            i -= (1 as $T);
        }
    } else if step > 0 { // ascending
        while i < stop {
            if !it(i) { return false; }
            // avoiding overflow. break if i + step > max_value
            if i > max_value - step { return true; }
            i += step;
        }
    } else { // descending
        while i > stop {
            if !it(i) { return false; }
            // avoiding underflow. break if i + step < min_value
            if i < min_value - step { return true; }
            i += step;
        }
    }
    match r {
        HalfOpen => return true,
        Closed => return (i != stop || it(i))
    }
}

#[inline]
///
/// Iterate through the range [`start`..`stop`) with a given step value.
///
/// Iterates through the range `[x_0, x_1, ..., x_n]` where
/// * `x_i == start + step*i`, and
/// * `n` is the greatest nonnegative integer such that `x_n < stop`
///
/// (If no such `n` exists, then the iteration range is empty.)
///
/// # Arguments
///
/// * `start` - lower bound, inclusive
/// * `stop` - higher bound, exclusive
///
/// # Examples
/// ~~~
/// let mut sum = 0;
/// for int::range(1, 5) |i| {
///     sum += i;
/// }
/// assert!(sum == 10);
/// ~~~
///
pub fn range_step(start: $T, stop: $T, step: $T, it: &fn($T) -> bool) -> bool {
    range_step_core(start, stop, step, HalfOpen, it)
}

#[inline]
///
/// Iterate through a range with a given step value.
///
/// Iterates through the range `[x_0, x_1, ..., x_n]` where
/// `x_i == start + step*i` and `x_n <= last < step + x_n`.
///
/// (If no such nonnegative integer `n` exists, then the iteration
///  range is empty.)
///
pub fn range_step_inclusive(start: $T, last: $T, step: $T, it: &fn($T) -> bool) -> bool {
    range_step_core(start, last, step, Closed, it)
}

impl Num for $T {}

#[cfg(not(test))]
impl Ord for $T {
    #[inline]
    fn lt(&self, other: &$T) -> bool { return (*self) < (*other); }
}

#[cfg(not(test))]
impl Eq for $T {
    #[inline]
    fn eq(&self, other: &$T) -> bool { return (*self) == (*other); }
}

impl Orderable for $T {
    #[inline]
    fn min(&self, other: &$T) -> $T {
        if *self < *other { *self } else { *other }
    }

    #[inline]
    fn max(&self, other: &$T) -> $T {
        if *self > *other { *self } else { *other }
    }

    #[inline]
    fn clamp(&self, mn: &$T, mx: &$T) -> $T {
        if *self > *mx { *mx } else
        if *self < *mn { *mn } else { *self }
    }
}

impl Zero for $T {
    #[inline]
    fn zero() -> $T { 0 }

    #[inline]
    fn is_zero(&self) -> bool { *self == 0 }
}

impl One for $T {
    #[inline]
    fn one() -> $T { 1 }
}

#[cfg(not(test))]
impl Add<$T,$T> for $T {
    #[inline]
    fn add(&self, other: &$T) -> $T { *self + *other }
}

#[cfg(not(test))]
impl Sub<$T,$T> for $T {
    #[inline]
    fn sub(&self, other: &$T) -> $T { *self - *other }
}

#[cfg(not(test))]
impl Mul<$T,$T> for $T {
    #[inline]
    fn mul(&self, other: &$T) -> $T { *self * *other }
}

#[cfg(not(test))]
impl Div<$T,$T> for $T {
    ///
    /// Integer division, truncated towards 0. As this behaviour reflects the underlying
    /// machine implementation it is more efficient than `Integer::div_floor`.
    ///
    /// # Examples
    ///
    /// ~~~
    /// assert!( 8 /  3 ==  2);
    /// assert!( 8 / -3 == -2);
    /// assert!(-8 /  3 == -2);
    /// assert!(-8 / -3 ==  2);

    /// assert!( 1 /  2 ==  0);
    /// assert!( 1 / -2 ==  0);
    /// assert!(-1 /  2 ==  0);
    /// assert!(-1 / -2 ==  0);
    /// ~~~
    ///
    #[inline]
    fn div(&self, other: &$T) -> $T { *self / *other }
}

#[cfg(not(test))]
impl Rem<$T,$T> for $T {
    ///
    /// Returns the integer remainder after division, satisfying:
    ///
    /// ~~~
    /// assert!((n / d) * d + (n % d) == n)
    /// ~~~
    ///
    /// # Examples
    ///
    /// ~~~
    /// assert!( 8 %  3 ==  2);
    /// assert!( 8 % -3 ==  2);
    /// assert!(-8 %  3 == -2);
    /// assert!(-8 % -3 == -2);

    /// assert!( 1 %  2 ==  1);
    /// assert!( 1 % -2 ==  1);
    /// assert!(-1 %  2 == -1);
    /// assert!(-1 % -2 == -1);
    /// ~~~
    ///
    #[inline]
    fn rem(&self, other: &$T) -> $T { *self % *other }
}

#[cfg(not(test))]
impl Neg<$T> for $T {
    #[inline]
    fn neg(&self) -> $T { -*self }
}

impl Signed for $T {
    /// Computes the absolute value
    #[inline]
    fn abs(&self) -> $T {
        if self.is_negative() { -*self } else { *self }
    }

    ///
    /// The positive difference of two numbers. Returns `0` if the number is less than or
    /// equal to `other`, otherwise the difference between`self` and `other` is returned.
    ///
    #[inline]
    fn abs_sub(&self, other: &$T) -> $T {
        if *self <= *other { 0 } else { *self - *other }
    }

    ///
    /// # Returns
    ///
    /// - `0` if the number is zero
    /// - `1` if the number is positive
    /// - `-1` if the number is negative
    ///
    #[inline]
    fn signum(&self) -> $T {
        match *self {
            n if n > 0 =>  1,
            0          =>  0,
            _          => -1,
        }
    }

    /// Returns true if the number is positive
    #[inline]
    fn is_positive(&self) -> bool { *self > 0 }

    /// Returns true if the number is negative
    #[inline]
    fn is_negative(&self) -> bool { *self < 0 }
}

impl Integer for $T {
    ///
    /// Floored integer division
    ///
    /// # Examples
    ///
    /// ~~~
    /// assert!(( 8).div_floor( 3) ==  2);
    /// assert!(( 8).div_floor(-3) == -3);
    /// assert!((-8).div_floor( 3) == -3);
    /// assert!((-8).div_floor(-3) ==  2);
    ///
    /// assert!(( 1).div_floor( 2) ==  0);
    /// assert!(( 1).div_floor(-2) == -1);
    /// assert!((-1).div_floor( 2) == -1);
    /// assert!((-1).div_floor(-2) ==  0);
    /// ~~~
    ///
    #[inline]
    fn div_floor(&self, other: &$T) -> $T {
        // Algorithm from [Daan Leijen. _Division and Modulus for Computer Scientists_,
        // December 2001](http://research.microsoft.com/pubs/151917/divmodnote-letter.pdf)
        match self.div_rem(other) {
            (d, r) if (r > 0 && *other < 0)
                   || (r < 0 && *other > 0) => d - 1,
            (d, _)                          => d,
        }
    }

    ///
    /// Integer modulo, satisfying:
    ///
    /// ~~~
    /// assert!(n.div_floor(d) * d + n.mod_floor(d) == n)
    /// ~~~
    ///
    /// # Examples
    ///
    /// ~~~
    /// assert!(( 8).mod_floor( 3) ==  2);
    /// assert!(( 8).mod_floor(-3) == -1);
    /// assert!((-8).mod_floor( 3) ==  1);
    /// assert!((-8).mod_floor(-3) == -2);
    ///
    /// assert!(( 1).mod_floor( 2) ==  1);
    /// assert!(( 1).mod_floor(-2) == -1);
    /// assert!((-1).mod_floor( 2) ==  1);
    /// assert!((-1).mod_floor(-2) == -1);
    /// ~~~
    ///
    #[inline]
    fn mod_floor(&self, other: &$T) -> $T {
        // Algorithm from [Daan Leijen. _Division and Modulus for Computer Scientists_,
        // December 2001](http://research.microsoft.com/pubs/151917/divmodnote-letter.pdf)
        match *self % *other {
            r if (r > 0 && *other < 0)
              || (r < 0 && *other > 0) => r + *other,
            r                          => r,
        }
    }

    /// Calculates `div_floor` and `mod_floor` simultaneously
    #[inline]
    fn div_mod_floor(&self, other: &$T) -> ($T,$T) {
        // Algorithm from [Daan Leijen. _Division and Modulus for Computer Scientists_,
        // December 2001](http://research.microsoft.com/pubs/151917/divmodnote-letter.pdf)
        match self.div_rem(other) {
            (d, r) if (r > 0 && *other < 0)
                   || (r < 0 && *other > 0) => (d - 1, r + *other),
            (d, r)                          => (d, r),
        }
    }

    /// Calculates `div` (`\`) and `rem` (`%`) simultaneously
    #[inline]
    fn div_rem(&self, other: &$T) -> ($T,$T) {
        (*self / *other, *self % *other)
    }

    ///
    /// Calculates the Greatest Common Divisor (GCD) of the number and `other`
    ///
    /// The result is always positive
    ///
    #[inline]
    fn gcd(&self, other: &$T) -> $T {
        // Use Euclid's algorithm
        let mut m = *self;
        let mut n = *other;
        while m != 0 {
            let temp = m;
            m = n % temp;
            n = temp;
        }
        n.abs()
    }

    ///
    /// Calculates the Lowest Common Multiple (LCM) of the number and `other`
    ///
    #[inline]
    fn lcm(&self, other: &$T) -> $T {
        ((*self * *other) / self.gcd(other)).abs() // should not have to recaluculate abs
    }

    /// Returns `true` if the number can be divided by `other` without leaving a remainder
    #[inline]
    fn is_multiple_of(&self, other: &$T) -> bool { *self % *other == 0 }

    /// Returns `true` if the number is divisible by `2`
    #[inline]
    fn is_even(&self) -> bool { self.is_multiple_of(&2) }

    /// Returns `true` if the number is not divisible by `2`
    #[inline]
    fn is_odd(&self) -> bool { !self.is_even() }
}

impl Bitwise for $T {}

#[cfg(not(test))]
impl BitOr<$T,$T> for $T {
    #[inline]
    fn bitor(&self, other: &$T) -> $T { *self | *other }
}

#[cfg(not(test))]
impl BitAnd<$T,$T> for $T {
    #[inline]
    fn bitand(&self, other: &$T) -> $T { *self & *other }
}

#[cfg(not(test))]
impl BitXor<$T,$T> for $T {
    #[inline]
    fn bitxor(&self, other: &$T) -> $T { *self ^ *other }
}

#[cfg(not(test))]
impl Shl<$T,$T> for $T {
    #[inline]
    fn shl(&self, other: &$T) -> $T { *self << *other }
}

#[cfg(not(test))]
impl Shr<$T,$T> for $T {
    #[inline]
    fn shr(&self, other: &$T) -> $T { *self >> *other }
}

#[cfg(not(test))]
impl Not<$T> for $T {
    #[inline]
    fn not(&self) -> $T { !*self }
}

impl Bounded for $T {
    #[inline]
    fn min_value() -> $T { min_value }

    #[inline]
    fn max_value() -> $T { max_value }
}

impl Int for $T {}

impl Primitive for $T {
    #[inline]
    fn bits(_: Option<$T>) -> uint { bits }

    #[inline]
    fn bytes(_: Option<$T>) -> uint { bits / 8 }
}

// String conversion functions and impl str -> num

/// Parse a string as a number in base 10.
#[inline]
pub fn from_str(s: &str) -> Option<$T> {
    strconv::from_str_common(s, 10u, true, false, false,
                         strconv::ExpNone, false, false)
}

/// Parse a string as a number in the given base.
#[inline]
pub fn from_str_radix(s: &str, radix: uint) -> Option<$T> {
    strconv::from_str_common(s, radix, true, false, false,
                         strconv::ExpNone, false, false)
}

/// Parse a byte slice as a number in the given base.
#[inline]
pub fn parse_bytes(buf: &[u8], radix: uint) -> Option<$T> {
    strconv::from_str_bytes_common(buf, radix, true, false, false,
                               strconv::ExpNone, false, false)
}

impl FromStr for $T {
    #[inline]
    fn from_str(s: &str) -> Option<$T> {
        from_str(s)
    }
}

impl FromStrRadix for $T {
    #[inline]
    fn from_str_radix(s: &str, radix: uint) -> Option<$T> {
        from_str_radix(s, radix)
    }
}

// String conversion functions and impl num -> str

/// Convert to a string as a byte slice in a given base.
#[inline]
pub fn to_str_bytes<U>(n: $T, radix: uint, f: &fn(v: &[u8]) -> U) -> U {
    // The radix can be as low as 2, so we need at least 64 characters for a
    // base 2 number, and then we need another for a possible '-' character.
    let mut buf = [0u8, ..65];
    let mut cur = 0;
    do strconv::int_to_str_bytes_common(n, radix, strconv::SignNeg) |i| {
        buf[cur] = i;
        cur += 1;
    }
    f(buf.slice(0, cur))
}

impl ToStr for $T {
    /// Convert to a string in base 10.
    #[inline]
    fn to_str(&self) -> ~str {
        self.to_str_radix(10)
    }
}

impl ToStrRadix for $T {
    /// Convert to a string in a given base.
    #[inline]
    fn to_str_radix(&self, radix: uint) -> ~str {
        let mut buf: ~[u8] = ~[];
        do strconv::int_to_str_bytes_common(*self, radix, strconv::SignNeg) |i| {
            buf.push(i);
        }
        // We know we generated valid utf-8, so we don't need to go through that
        // check.
        unsafe { str::raw::from_bytes_owned(buf) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prelude::*;

    use int;
    use i16;
    use i32;
    use i64;
    use i8;
    use num;
    use sys;

    #[test]
    fn test_num() {
        num::test_num(10 as $T, 2 as $T);
    }

    #[test]
    fn test_orderable() {
        assert_eq!((1 as $T).min(&(2 as $T)), 1 as $T);
        assert_eq!((2 as $T).min(&(1 as $T)), 1 as $T);
        assert_eq!((1 as $T).max(&(2 as $T)), 2 as $T);
        assert_eq!((2 as $T).max(&(1 as $T)), 2 as $T);
        assert_eq!((1 as $T).clamp(&(2 as $T), &(4 as $T)), 2 as $T);
        assert_eq!((8 as $T).clamp(&(2 as $T), &(4 as $T)), 4 as $T);
        assert_eq!((3 as $T).clamp(&(2 as $T), &(4 as $T)), 3 as $T);
    }

    #[test]
    pub fn test_abs() {
        assert_eq!((1 as $T).abs(), 1 as $T);
        assert_eq!((0 as $T).abs(), 0 as $T);
        assert_eq!((-1 as $T).abs(), 1 as $T);
    }

    #[test]
    fn test_abs_sub() {
        assert_eq!((-1 as $T).abs_sub(&(1 as $T)), 0 as $T);
        assert_eq!((1 as $T).abs_sub(&(1 as $T)), 0 as $T);
        assert_eq!((1 as $T).abs_sub(&(0 as $T)), 1 as $T);
        assert_eq!((1 as $T).abs_sub(&(-1 as $T)), 2 as $T);
    }

    #[test]
    fn test_signum() {
        assert_eq!((1 as $T).signum(), 1 as $T);
        assert_eq!((0 as $T).signum(), 0 as $T);
        assert_eq!((-0 as $T).signum(), 0 as $T);
        assert_eq!((-1 as $T).signum(), -1 as $T);
    }

    #[test]
    fn test_is_positive() {
        assert!((1 as $T).is_positive());
        assert!(!(0 as $T).is_positive());
        assert!(!(-0 as $T).is_positive());
        assert!(!(-1 as $T).is_positive());
    }

    #[test]
    fn test_is_negative() {
        assert!(!(1 as $T).is_negative());
        assert!(!(0 as $T).is_negative());
        assert!(!(-0 as $T).is_negative());
        assert!((-1 as $T).is_negative());
    }

    ///
    /// Checks that the division rule holds for:
    ///
    /// - `n`: numerator (dividend)
    /// - `d`: denominator (divisor)
    /// - `qr`: quotient and remainder
    ///
    #[cfg(test)]
    fn test_division_rule((n,d): ($T,$T), (q,r): ($T,$T)) {
        assert_eq!(d * q + r, n);
    }

    #[test]
    fn test_div_rem() {
        fn test_nd_dr(nd: ($T,$T), qr: ($T,$T)) {
            let (n,d) = nd;
            let separate_div_rem = (n / d, n % d);
            let combined_div_rem = n.div_rem(&d);

            assert_eq!(separate_div_rem, qr);
            assert_eq!(combined_div_rem, qr);

            test_division_rule(nd, separate_div_rem);
            test_division_rule(nd, combined_div_rem);
        }

        test_nd_dr(( 8,  3), ( 2,  2));
        test_nd_dr(( 8, -3), (-2,  2));
        test_nd_dr((-8,  3), (-2, -2));
        test_nd_dr((-8, -3), ( 2, -2));

        test_nd_dr(( 1,  2), ( 0,  1));
        test_nd_dr(( 1, -2), ( 0,  1));
        test_nd_dr((-1,  2), ( 0, -1));
        test_nd_dr((-1, -2), ( 0, -1));
    }

    #[test]
    fn test_div_mod_floor() {
        fn test_nd_dm(nd: ($T,$T), dm: ($T,$T)) {
            let (n,d) = nd;
            let separate_div_mod_floor = (n.div_floor(&d), n.mod_floor(&d));
            let combined_div_mod_floor = n.div_mod_floor(&d);

            assert_eq!(separate_div_mod_floor, dm);
            assert_eq!(combined_div_mod_floor, dm);

            test_division_rule(nd, separate_div_mod_floor);
            test_division_rule(nd, combined_div_mod_floor);
        }

        test_nd_dm(( 8,  3), ( 2,  2));
        test_nd_dm(( 8, -3), (-3, -1));
        test_nd_dm((-8,  3), (-3,  1));
        test_nd_dm((-8, -3), ( 2, -2));

        test_nd_dm(( 1,  2), ( 0,  1));
        test_nd_dm(( 1, -2), (-1, -1));
        test_nd_dm((-1,  2), (-1,  1));
        test_nd_dm((-1, -2), ( 0, -1));
    }

    #[test]
    fn test_gcd() {
        assert_eq!((10 as $T).gcd(&2), 2 as $T);
        assert_eq!((10 as $T).gcd(&3), 1 as $T);
        assert_eq!((0 as $T).gcd(&3), 3 as $T);
        assert_eq!((3 as $T).gcd(&3), 3 as $T);
        assert_eq!((56 as $T).gcd(&42), 14 as $T);
        assert_eq!((3 as $T).gcd(&-3), 3 as $T);
        assert_eq!((-6 as $T).gcd(&3), 3 as $T);
        assert_eq!((-4 as $T).gcd(&-2), 2 as $T);
    }

    #[test]
    fn test_lcm() {
        assert_eq!((1 as $T).lcm(&0), 0 as $T);
        assert_eq!((0 as $T).lcm(&1), 0 as $T);
        assert_eq!((1 as $T).lcm(&1), 1 as $T);
        assert_eq!((-1 as $T).lcm(&1), 1 as $T);
        assert_eq!((1 as $T).lcm(&-1), 1 as $T);
        assert_eq!((-1 as $T).lcm(&-1), 1 as $T);
        assert_eq!((8 as $T).lcm(&9), 72 as $T);
        assert_eq!((11 as $T).lcm(&5), 55 as $T);
    }

    #[test]
    fn test_bitwise() {
        assert_eq!(0b1110 as $T, (0b1100 as $T).bitor(&(0b1010 as $T)));
        assert_eq!(0b1000 as $T, (0b1100 as $T).bitand(&(0b1010 as $T)));
        assert_eq!(0b0110 as $T, (0b1100 as $T).bitxor(&(0b1010 as $T)));
        assert_eq!(0b1110 as $T, (0b0111 as $T).shl(&(1 as $T)));
        assert_eq!(0b0111 as $T, (0b1110 as $T).shr(&(1 as $T)));
        assert_eq!(-(0b11 as $T) - (1 as $T), (0b11 as $T).not());
    }

    #[test]
    fn test_multiple_of() {
        assert!((6 as $T).is_multiple_of(&(6 as $T)));
        assert!((6 as $T).is_multiple_of(&(3 as $T)));
        assert!((6 as $T).is_multiple_of(&(1 as $T)));
        assert!((-8 as $T).is_multiple_of(&(4 as $T)));
        assert!((8 as $T).is_multiple_of(&(-1 as $T)));
        assert!((-8 as $T).is_multiple_of(&(-2 as $T)));
    }

    #[test]
    fn test_even() {
        assert_eq!((-4 as $T).is_even(), true);
        assert_eq!((-3 as $T).is_even(), false);
        assert_eq!((-2 as $T).is_even(), true);
        assert_eq!((-1 as $T).is_even(), false);
        assert_eq!((0 as $T).is_even(), true);
        assert_eq!((1 as $T).is_even(), false);
        assert_eq!((2 as $T).is_even(), true);
        assert_eq!((3 as $T).is_even(), false);
        assert_eq!((4 as $T).is_even(), true);
    }

    #[test]
    fn test_odd() {
        assert_eq!((-4 as $T).is_odd(), false);
        assert_eq!((-3 as $T).is_odd(), true);
        assert_eq!((-2 as $T).is_odd(), false);
        assert_eq!((-1 as $T).is_odd(), true);
        assert_eq!((0 as $T).is_odd(), false);
        assert_eq!((1 as $T).is_odd(), true);
        assert_eq!((2 as $T).is_odd(), false);
        assert_eq!((3 as $T).is_odd(), true);
        assert_eq!((4 as $T).is_odd(), false);
    }

    #[test]
    fn test_bitcount() {
        assert_eq!((0b010101 as $T).population_count(), 3);
    }

    #[test]
    fn test_primitive() {
        let none: Option<$T> = None;
        assert_eq!(Primitive::bits(none), sys::size_of::<$T>() * 8);
        assert_eq!(Primitive::bytes(none), sys::size_of::<$T>());
    }

    #[test]
    fn test_from_str() {
        assert_eq!(from_str("0"), Some(0 as $T));
        assert_eq!(from_str("3"), Some(3 as $T));
        assert_eq!(from_str("10"), Some(10 as $T));
        assert_eq!(i32::from_str("123456789"), Some(123456789 as i32));
        assert_eq!(from_str("00100"), Some(100 as $T));

        assert_eq!(from_str("-1"), Some(-1 as $T));
        assert_eq!(from_str("-3"), Some(-3 as $T));
        assert_eq!(from_str("-10"), Some(-10 as $T));
        assert_eq!(i32::from_str("-123456789"), Some(-123456789 as i32));
        assert_eq!(from_str("-00100"), Some(-100 as $T));

        assert!(from_str(" ").is_none());
        assert!(from_str("x").is_none());
    }

    #[test]
    fn test_parse_bytes() {
        use str::StrSlice;
        assert_eq!(parse_bytes("123".as_bytes(), 10u), Some(123 as $T));
        assert_eq!(parse_bytes("1001".as_bytes(), 2u), Some(9 as $T));
        assert_eq!(parse_bytes("123".as_bytes(), 8u), Some(83 as $T));
        assert_eq!(i32::parse_bytes("123".as_bytes(), 16u), Some(291 as i32));
        assert_eq!(i32::parse_bytes("ffff".as_bytes(), 16u), Some(65535 as i32));
        assert_eq!(i32::parse_bytes("FFFF".as_bytes(), 16u), Some(65535 as i32));
        assert_eq!(parse_bytes("z".as_bytes(), 36u), Some(35 as $T));
        assert_eq!(parse_bytes("Z".as_bytes(), 36u), Some(35 as $T));

        assert_eq!(parse_bytes("-123".as_bytes(), 10u), Some(-123 as $T));
        assert_eq!(parse_bytes("-1001".as_bytes(), 2u), Some(-9 as $T));
        assert_eq!(parse_bytes("-123".as_bytes(), 8u), Some(-83 as $T));
        assert_eq!(i32::parse_bytes("-123".as_bytes(), 16u), Some(-291 as i32));
        assert_eq!(i32::parse_bytes("-ffff".as_bytes(), 16u), Some(-65535 as i32));
        assert_eq!(i32::parse_bytes("-FFFF".as_bytes(), 16u), Some(-65535 as i32));
        assert_eq!(parse_bytes("-z".as_bytes(), 36u), Some(-35 as $T));
        assert_eq!(parse_bytes("-Z".as_bytes(), 36u), Some(-35 as $T));

        assert!(parse_bytes("Z".as_bytes(), 35u).is_none());
        assert!(parse_bytes("-9".as_bytes(), 2u).is_none());
    }

    #[test]
    fn test_to_str() {
        assert_eq!((0 as $T).to_str_radix(10u), ~"0");
        assert_eq!((1 as $T).to_str_radix(10u), ~"1");
        assert_eq!((-1 as $T).to_str_radix(10u), ~"-1");
        assert_eq!((127 as $T).to_str_radix(16u), ~"7f");
        assert_eq!((100 as $T).to_str_radix(10u), ~"100");

    }

    #[test]
    fn test_int_to_str_overflow() {
        let mut i8_val: i8 = 127_i8;
        assert_eq!(i8_val.to_str(), ~"127");

        i8_val += 1 as i8;
        assert_eq!(i8_val.to_str(), ~"-128");

        let mut i16_val: i16 = 32_767_i16;
        assert_eq!(i16_val.to_str(), ~"32767");

        i16_val += 1 as i16;
        assert_eq!(i16_val.to_str(), ~"-32768");

        let mut i32_val: i32 = 2_147_483_647_i32;
        assert_eq!(i32_val.to_str(), ~"2147483647");

        i32_val += 1 as i32;
        assert_eq!(i32_val.to_str(), ~"-2147483648");

        let mut i64_val: i64 = 9_223_372_036_854_775_807_i64;
        assert_eq!(i64_val.to_str(), ~"9223372036854775807");

        i64_val += 1 as i64;
        assert_eq!(i64_val.to_str(), ~"-9223372036854775808");
    }

    #[test]
    fn test_int_from_str_overflow() {
        let mut i8_val: i8 = 127_i8;
        assert_eq!(i8::from_str("127"), Some(i8_val));
        assert!(i8::from_str("128").is_none());

        i8_val += 1 as i8;
        assert_eq!(i8::from_str("-128"), Some(i8_val));
        assert!(i8::from_str("-129").is_none());

        let mut i16_val: i16 = 32_767_i16;
        assert_eq!(i16::from_str("32767"), Some(i16_val));
        assert!(i16::from_str("32768").is_none());

        i16_val += 1 as i16;
        assert_eq!(i16::from_str("-32768"), Some(i16_val));
        assert!(i16::from_str("-32769").is_none());

        let mut i32_val: i32 = 2_147_483_647_i32;
        assert_eq!(i32::from_str("2147483647"), Some(i32_val));
        assert!(i32::from_str("2147483648").is_none());

        i32_val += 1 as i32;
        assert_eq!(i32::from_str("-2147483648"), Some(i32_val));
        assert!(i32::from_str("-2147483649").is_none());

        let mut i64_val: i64 = 9_223_372_036_854_775_807_i64;
        assert_eq!(i64::from_str("9223372036854775807"), Some(i64_val));
        assert!(i64::from_str("9223372036854775808").is_none());

        i64_val += 1 as i64;
        assert_eq!(i64::from_str("-9223372036854775808"), Some(i64_val));
        assert!(i64::from_str("-9223372036854775809").is_none());
    }

    #[test]
    fn test_ranges() {
        let mut l = ~[];

        do range_step(20,26,2) |i| {
            l.push(i);
            true
        };
        do range_step(36,30,-2) |i| {
            l.push(i);
            true
        };
        do range_step(max_value - 2, max_value, 2) |i| {
            l.push(i);
            true
        };
        do range_step(max_value - 3, max_value, 2) |i| {
            l.push(i);
            true
        };
        do range_step(min_value + 2, min_value, -2) |i| {
            l.push(i);
            true
        };
        do range_step(min_value + 3, min_value, -2) |i| {
            l.push(i);
            true
        };
        assert_eq!(l, ~[20,22,24,
                        36,34,32,
                        max_value-2,
                        max_value-3,max_value-1,
                        min_value+2,
                        min_value+3,min_value+1]);

        // None of the `fail`s should execute.
        do range_step(10,0,1) |_i| {
            fail!(~"unreachable");
        };
        do range_step(0,10,-1) |_i| {
            fail!(~"unreachable");
        };
    }

    #[test]
    #[should_fail]
    fn test_range_step_zero_step() {
        do range_step(0,10,0) |_i| { true };
    }

    #[test]
    fn test_signed_checked_div() {
        assert_eq!(10i.checked_div(&2), Some(5));
        assert_eq!(5i.checked_div(&0), None);
        assert_eq!(int::min_value.checked_div(&-1), None);
    }
}

}))
