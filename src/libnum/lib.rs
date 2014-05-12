// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(macro_rules)]

#![crate_id = "num#0.11.0-pre"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://static.rust-lang.org/doc/master")]

#![deny(deprecated_owned_vector)]

extern crate rand;

pub mod bigint;
pub mod rational;
pub mod complex;

pub trait Integer: Num + Ord
                 + Div<Self, Self>
                 + Rem<Self, Self> {
    /// Simultaneous truncated integer division and modulus
    #[inline]
    fn div_rem(&self, other: &Self) -> (Self, Self) {
        (*self / *other, *self % *other)
    }

    /// Floored integer division
    ///
    /// # Examples
    ///
    /// ~~~
    /// # use num::Integer;
    /// assert!(( 8i).div_floor(& 3) ==  2);
    /// assert!(( 8i).div_floor(&-3) == -3);
    /// assert!((-8i).div_floor(& 3) == -3);
    /// assert!((-8i).div_floor(&-3) ==  2);
    ///
    /// assert!(( 1i).div_floor(& 2) ==  0);
    /// assert!(( 1i).div_floor(&-2) == -1);
    /// assert!((-1i).div_floor(& 2) == -1);
    /// assert!((-1i).div_floor(&-2) ==  0);
    /// ~~~
    fn div_floor(&self, other: &Self) -> Self;

    /// Floored integer modulo, satisfying:
    ///
    /// ~~~
    /// # use num::Integer;
    /// # let n = 1i; let d = 1i;
    /// assert!(n.div_floor(&d) * d + n.mod_floor(&d) == n)
    /// ~~~
    ///
    /// # Examples
    ///
    /// ~~~
    /// # use num::Integer;
    /// assert!(( 8i).mod_floor(& 3) ==  2);
    /// assert!(( 8i).mod_floor(&-3) == -1);
    /// assert!((-8i).mod_floor(& 3) ==  1);
    /// assert!((-8i).mod_floor(&-3) == -2);
    ///
    /// assert!(( 1i).mod_floor(& 2) ==  1);
    /// assert!(( 1i).mod_floor(&-2) == -1);
    /// assert!((-1i).mod_floor(& 2) ==  1);
    /// assert!((-1i).mod_floor(&-2) == -1);
    /// ~~~
    fn mod_floor(&self, other: &Self) -> Self;

    /// Simultaneous floored integer division and modulus
    fn div_mod_floor(&self, other: &Self) -> (Self, Self) {
        (self.div_floor(other), self.mod_floor(other))
    }

    /// Greatest Common Divisor (GCD)
    fn gcd(&self, other: &Self) -> Self;

    /// Lowest Common Multiple (LCM)
    fn lcm(&self, other: &Self) -> Self;

    /// Returns `true` if `other` divides evenly into `self`
    fn divides(&self, other: &Self) -> bool;

    /// Returns `true` if the number is even
    fn is_even(&self) -> bool;

    /// Returns `true` if the number is odd
    fn is_odd(&self) -> bool;
}

/// Simultaneous integer division and modulus
#[inline] pub fn div_rem<T: Integer>(x: T, y: T) -> (T, T) { x.div_rem(&y) }
/// Floored integer division
#[inline] pub fn div_floor<T: Integer>(x: T, y: T) -> T { x.div_floor(&y) }
/// Floored integer modulus
#[inline] pub fn mod_floor<T: Integer>(x: T, y: T) -> T { x.mod_floor(&y) }
/// Simultaneous floored integer division and modulus
#[inline] pub fn div_mod_floor<T: Integer>(x: T, y: T) -> (T, T) { x.div_mod_floor(&y) }

/// Calculates the Greatest Common Divisor (GCD) of the number and `other`. The
/// result is always positive.
#[inline(always)] pub fn gcd<T: Integer>(x: T, y: T) -> T { x.gcd(&y) }
/// Calculates the Lowest Common Multiple (LCM) of the number and `other`.
#[inline(always)] pub fn lcm<T: Integer>(x: T, y: T) -> T { x.lcm(&y) }

macro_rules! impl_integer_for_int {
    ($T:ty, $test_mod:ident) => (
        impl Integer for $T {
            /// Floored integer division
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

            /// Floored integer modulo
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

            /// Calculates the Greatest Common Divisor (GCD) of the number and
            /// `other`. The result is always positive.
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

            /// Calculates the Lowest Common Multiple (LCM) of the number and
            /// `other`.
            #[inline]
            fn lcm(&self, other: &$T) -> $T {
                // should not have to recalculate abs
                ((*self * *other) / self.gcd(other)).abs()
            }

            /// Returns `true` if the number can be divided by `other` without
            /// leaving a remainder
            #[inline]
            fn divides(&self, other: &$T) -> bool { *self % *other == 0 }

            /// Returns `true` if the number is divisible by `2`
            #[inline]
            fn is_even(&self) -> bool { self & 1 == 0 }

            /// Returns `true` if the number is not divisible by `2`
            #[inline]
            fn is_odd(&self) -> bool { !self.is_even() }
        }

        #[cfg(test)]
        mod $test_mod {
            use Integer;

            /// Checks that the division rule holds for:
            ///
            /// - `n`: numerator (dividend)
            /// - `d`: denominator (divisor)
            /// - `qr`: quotient and remainder
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
        }
    )
}

impl_integer_for_int!(i8,   test_integer_i8)
impl_integer_for_int!(i16,  test_integer_i16)
impl_integer_for_int!(i32,  test_integer_i32)
impl_integer_for_int!(i64,  test_integer_i64)
impl_integer_for_int!(int,  test_integer_int)

macro_rules! impl_integer_for_uint {
    ($T:ty, $test_mod:ident) => (
        impl Integer for $T {
            /// Unsigned integer division. Returns the same result as `div` (`/`).
            #[inline]
            fn div_floor(&self, other: &$T) -> $T { *self / *other }

            /// Unsigned integer modulo operation. Returns the same result as `rem` (`%`).
            #[inline]
            fn mod_floor(&self, other: &$T) -> $T { *self % *other }

            /// Calculates the Greatest Common Divisor (GCD) of the number and `other`
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
                n
            }

            /// Calculates the Lowest Common Multiple (LCM) of the number and `other`
            #[inline]
            fn lcm(&self, other: &$T) -> $T {
                (*self * *other) / self.gcd(other)
            }

            /// Returns `true` if the number can be divided by `other` without leaving a remainder
            #[inline]
            fn divides(&self, other: &$T) -> bool { *self % *other == 0 }

            /// Returns `true` if the number is divisible by `2`
            #[inline]
            fn is_even(&self) -> bool { self & 1 == 0 }

            /// Returns `true` if the number is not divisible by `2`
            #[inline]
            fn is_odd(&self) -> bool { !self.is_even() }
        }

        #[cfg(test)]
        mod $test_mod {
            use Integer;

            #[test]
            fn test_div_mod_floor() {
                assert_eq!((10 as $T).div_floor(&(3 as $T)), 3 as $T);
                assert_eq!((10 as $T).mod_floor(&(3 as $T)), 1 as $T);
                assert_eq!((10 as $T).div_mod_floor(&(3 as $T)), (3 as $T, 1 as $T));
                assert_eq!((5 as $T).div_floor(&(5 as $T)), 1 as $T);
                assert_eq!((5 as $T).mod_floor(&(5 as $T)), 0 as $T);
                assert_eq!((5 as $T).div_mod_floor(&(5 as $T)), (1 as $T, 0 as $T));
                assert_eq!((3 as $T).div_floor(&(7 as $T)), 0 as $T);
                assert_eq!((3 as $T).mod_floor(&(7 as $T)), 3 as $T);
                assert_eq!((3 as $T).div_mod_floor(&(7 as $T)), (0 as $T, 3 as $T));
            }

            #[test]
            fn test_gcd() {
                assert_eq!((10 as $T).gcd(&2), 2 as $T);
                assert_eq!((10 as $T).gcd(&3), 1 as $T);
                assert_eq!((0 as $T).gcd(&3), 3 as $T);
                assert_eq!((3 as $T).gcd(&3), 3 as $T);
                assert_eq!((56 as $T).gcd(&42), 14 as $T);
            }

            #[test]
            fn test_lcm() {
                assert_eq!((1 as $T).lcm(&0), 0 as $T);
                assert_eq!((0 as $T).lcm(&1), 0 as $T);
                assert_eq!((1 as $T).lcm(&1), 1 as $T);
                assert_eq!((8 as $T).lcm(&9), 72 as $T);
                assert_eq!((11 as $T).lcm(&5), 55 as $T);
                assert_eq!((99 as $T).lcm(&17), 1683 as $T);
            }

            #[test]
            fn test_divides() {
                assert!((6 as $T).divides(&(6 as $T)));
                assert!((6 as $T).divides(&(3 as $T)));
                assert!((6 as $T).divides(&(1 as $T)));
            }

            #[test]
            fn test_even() {
                assert_eq!((0 as $T).is_even(), true);
                assert_eq!((1 as $T).is_even(), false);
                assert_eq!((2 as $T).is_even(), true);
                assert_eq!((3 as $T).is_even(), false);
                assert_eq!((4 as $T).is_even(), true);
            }

            #[test]
            fn test_odd() {
                assert_eq!((0 as $T).is_odd(), false);
                assert_eq!((1 as $T).is_odd(), true);
                assert_eq!((2 as $T).is_odd(), false);
                assert_eq!((3 as $T).is_odd(), true);
                assert_eq!((4 as $T).is_odd(), false);
            }
        }
    )
}

impl_integer_for_uint!(u8,   test_integer_u8)
impl_integer_for_uint!(u16,  test_integer_u16)
impl_integer_for_uint!(u32,  test_integer_u32)
impl_integer_for_uint!(u64,  test_integer_u64)
impl_integer_for_uint!(uint, test_integer_uint)
