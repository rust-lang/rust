// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


//! Rational numbers

use core::num::{Zero,One,ToStrRadix,FromStrRadix,Round};
use core::from_str::FromStr;
use core::to_str::ToStr;
use core::prelude::*;
use core::cmp::TotalEq;
use super::bigint::BigInt;

/// Represents the ratio between 2 numbers.
#[deriving(Clone)]
pub struct Ratio<T> {
    numer: T,
    denom: T
}

/// Alias for a `Ratio` of machine-sized integers.
pub type Rational = Ratio<int>;
pub type Rational32 = Ratio<i32>;
pub type Rational64 = Ratio<i64>;

/// Alias for arbitrary precision rationals.
pub type BigRational = Ratio<BigInt>;

impl<T: Copy + Num + Ord>
    Ratio<T> {
    /// Create a ratio representing the integer `t`.
    #[inline(always)]
    pub fn from_integer(t: T) -> Ratio<T> {
        Ratio::new_raw(t, One::one())
    }

    /// Create a ratio without checking for `denom == 0` or reducing.
    #[inline(always)]
    pub fn new_raw(numer: T, denom: T) -> Ratio<T> {
        Ratio { numer: numer, denom: denom }
    }

    // Create a new Ratio. Fails if `denom == 0`.
    #[inline(always)]
    pub fn new(numer: T, denom: T) -> Ratio<T> {
        if denom == Zero::zero() {
            fail!(~"denominator == 0");
        }
        let mut ret = Ratio::new_raw(numer, denom);
        ret.reduce();
        ret
    }

    /// Put self into lowest terms, with denom > 0.
    fn reduce(&mut self) {
        let g : T = gcd(self.numer, self.denom);

        self.numer /= g;
        self.denom /= g;

        // keep denom positive!
        if self.denom < Zero::zero() {
            self.numer = -self.numer;
            self.denom = -self.denom;
        }
    }
    /// Return a `reduce`d copy of self.
    fn reduced(&self) -> Ratio<T> {
        let mut ret = copy *self;
        ret.reduce();
        ret
    }
}

/**
Compute the greatest common divisor of two numbers, via Euclid's algorithm.

The result can be negative.
*/
#[inline]
pub fn gcd_raw<T: Num>(n: T, m: T) -> T {
    let mut m = m, n = n;
    while m != Zero::zero() {
        let temp = m;
        m = n % temp;
        n = temp;
    }
    n
}

/**
Compute the greatest common divisor of two numbers, via Euclid's algorithm.

The result is always positive.
*/
#[inline]
pub fn gcd<T: Num + Ord>(n: T, m: T) -> T {
    let g = gcd_raw(n, m);
    if g < Zero::zero() { -g }
    else { g }
}

/* Comparisons */

// comparing a/b and c/d is the same as comparing a*d and b*c, so we
// abstract that pattern. The following macro takes a trait and either
// a comma-separated list of "method name -> return value" or just
// "method name" (return value is bool in that case)
macro_rules! cmp_impl {
    (impl $imp:ident, $($method:ident),+) => {
        cmp_impl!(impl $imp, $($method -> bool),+)
    };
    // return something other than a Ratio<T>
    (impl $imp:ident, $($method:ident -> $res:ty),+) => {
        impl<T: Mul<T,T> + $imp> $imp for Ratio<T> {
            $(
                #[inline]
                fn $method(&self, other: &Ratio<T>) -> $res {
                    (self.numer * other.denom). $method (&(self.denom*other.numer))
                }
            )+
        }
    };
}
cmp_impl!(impl Eq, eq, ne)
cmp_impl!(impl TotalEq, equals)
cmp_impl!(impl Ord, lt, gt, le, ge)
cmp_impl!(impl TotalOrd, cmp -> cmp::Ordering)

/* Arithmetic */
// a/b * c/d = (a*c)/(b*d)
impl<T: Copy + Num + Ord>
    Mul<Ratio<T>,Ratio<T>> for Ratio<T> {
    #[inline]
    fn mul(&self, rhs: &Ratio<T>) -> Ratio<T> {
        Ratio::new(self.numer * rhs.numer, self.denom * rhs.denom)
    }
}

// (a/b) / (c/d) = (a*d)/(b*c)
impl<T: Copy + Num + Ord>
    Quot<Ratio<T>,Ratio<T>> for Ratio<T> {
    #[inline]
    fn quot(&self, rhs: &Ratio<T>) -> Ratio<T> {
        Ratio::new(self.numer * rhs.denom, self.denom * rhs.numer)
    }
}

// Abstracts the a/b `op` c/d = (a*d `op` b*d) / (b*d) pattern
macro_rules! arith_impl {
    (impl $imp:ident, $method:ident) => {
        impl<T: Copy + Num + Ord>
            $imp<Ratio<T>,Ratio<T>> for Ratio<T> {
            #[inline]
            fn $method(&self, rhs: &Ratio<T>) -> Ratio<T> {
                Ratio::new((self.numer * rhs.denom).$method(&(self.denom * rhs.numer)),
                           self.denom * rhs.denom)
            }
        }
    }
}

// a/b + c/d = (a*d + b*c)/(b*d
arith_impl!(impl Add, add)

// a/b - c/d = (a*d - b*c)/(b*d)
arith_impl!(impl Sub, sub)

// a/b % c/d = (a*d % b*c)/(b*d)
arith_impl!(impl Rem, rem)

impl<T: Copy + Num + Ord>
    Neg<Ratio<T>> for Ratio<T> {
    #[inline]
    fn neg(&self) -> Ratio<T> {
        Ratio::new_raw(-self.numer, self.denom)
    }
}

/* Constants */
impl<T: Copy + Num + Ord>
    Zero for Ratio<T> {
    #[inline]
    fn zero() -> Ratio<T> {
        Ratio::new_raw(Zero::zero(), One::one())
    }
}

impl<T: Copy + Num + Ord>
    One for Ratio<T> {
    #[inline]
    fn one() -> Ratio<T> {
        Ratio::new_raw(One::one(), One::one())
    }
}

/* Utils */
impl<T: Copy + Num + Ord>
    Round for Ratio<T> {
    fn round(&self, mode: num::RoundMode) -> Ratio<T> {
        match mode {
            num::RoundUp => { self.ceil() }
            num::RoundDown => { self.floor()}
            num::RoundToZero => { Ratio::from_integer(self.numer / self.denom) }
            num::RoundFromZero => {
                if *self < Zero::zero() {
                    Ratio::from_integer((self.numer - self.denom + One::one()) / self.denom)
                } else {
                    Ratio::from_integer((self.numer + self.denom - One::one()) / self.denom)
                }
            }
        }
    }

    fn floor(&self) -> Ratio<T> {
        if *self < Zero::zero() {
            Ratio::from_integer((self.numer - self.denom + One::one()) / self.denom)
        } else {
            Ratio::from_integer(self.numer / self.denom)
        }
    }
    fn ceil(&self) -> Ratio<T> {
        if *self < Zero::zero() {
            Ratio::from_integer(self.numer / self.denom)
        } else {
            Ratio::from_integer((self.numer + self.denom - One::one()) / self.denom)
        }
    }
    fn fract(&self) -> Ratio<T> {
        Ratio::new_raw(self.numer % self.denom, self.denom)
    }
}


/* String conversions */
impl<T: ToStr> ToStr for Ratio<T> {
    /// Renders as `numer/denom`.
    fn to_str(&self) -> ~str {
        fmt!("%s/%s", self.numer.to_str(), self.denom.to_str())
    }
}
impl<T: ToStrRadix> ToStrRadix for Ratio<T> {
    /// Renders as `numer/denom` where the numbers are in base `radix`.
    fn to_str_radix(&self, radix: uint) -> ~str {
        fmt!("%s/%s", self.numer.to_str_radix(radix), self.denom.to_str_radix(radix))
    }
}

impl<T: FromStr + Copy + Num + Ord>
    FromStr for Ratio<T> {
    /// Parses `numer/denom`.
    fn from_str(s: &str) -> Option<Ratio<T>> {
        let split = vec::build(|push| {
            for str::each_splitn_char(s, '/', 1) |s| {
                push(s.to_owned());
            }
        });
        if split.len() < 2 { return None; }
        do FromStr::from_str(split[0]).chain |a| {
            do FromStr::from_str(split[1]).chain |b| {
                Some(Ratio::new(a,b))
            }
        }
    }
}
impl<T: FromStrRadix + Copy + Num + Ord>
    FromStrRadix for Ratio<T> {
    /// Parses `numer/denom` where the numbers are in base `radix`.
    fn from_str_radix(s: &str, radix: uint) -> Option<Ratio<T>> {
        let split = vec::build(|push| {
            for str::each_splitn_char(s, '/', 1) |s| {
                push(s.to_owned());
            }
        });
        if split.len() < 2 { None }
        else {
            do FromStrRadix::from_str_radix(split[0], radix).chain |a| {
                do FromStrRadix::from_str_radix(split[1], radix).chain |b| {
                    Some(Ratio::new(a,b))
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use core::prelude::*;
    use super::*;
    use core::num::{Zero,One,FromStrRadix};
    use core::from_str::FromStr;

    pub static _0 : Rational = Ratio { numer: 0, denom: 1};
    pub static _1 : Rational = Ratio { numer: 1, denom: 1};
    pub static _2: Rational = Ratio { numer: 2, denom: 1};
    pub static _1_2: Rational = Ratio { numer: 1, denom: 2};
    pub static _3_2: Rational = Ratio { numer: 3, denom: 2};
    pub static _neg1_2: Rational =  Ratio { numer: -1, denom: 2};

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(10,2),2);
        assert_eq!(gcd(10,3),1);
        assert_eq!(gcd(0,3),3);
        assert_eq!(gcd(3,3),3);

        assert_eq!(gcd(3,-3), 3);
        assert_eq!(gcd(-6,3), 3);
        assert_eq!(gcd(-4,-2), 2);
    }

    #[test]
    fn test_test_constants() {
        // check our constants are what Ratio::new etc. would make.
        assert_eq!(_0, Zero::zero());
        assert_eq!(_1, One::one());
        assert_eq!(_2, Ratio::from_integer(2));
        assert_eq!(_1_2, Ratio::new(1,2));
        assert_eq!(_3_2, Ratio::new(3,2));
        assert_eq!(_neg1_2, Ratio::new(-1,2));
    }

    #[test]
    fn test_new_reduce() {
        let one22 = Ratio::new(2i,2);

        assert_eq!(one22, One::one());
    }
    #[test]
    #[should_fail]
    fn test_new_zero() {
        let _a = Ratio::new(1,0);
    }


    #[test]
    fn test_cmp() {
        assert!(_0 == _0 && _1 == _1);
        assert!(_0 != _1 && _1 != _0);
        assert!(_0 < _1 && !(_1 < _0));
        assert!(_1 > _0 && !(_0 > _1));

        assert!(_0 <= _0 && _1 <= _1);
        assert!(_0 <= _1 && !(_1 <= _0));

        assert!(_0 >= _0 && _1 >= _1);
        assert!(_1 >= _0 && !(_0 >= _1));
    }


    mod arith {
        use super::*;
        use super::super::*;


        #[test]
        fn test_add() {
            assert_eq!(_1 + _1_2, _3_2);
            assert_eq!(_1 + _1, _2);
            assert_eq!(_1_2 + _3_2, _2);
            assert_eq!(_1_2 + _neg1_2, _0);
        }

        #[test]
        fn test_sub() {
            assert_eq!(_1 - _1_2, _1_2);
            assert_eq!(_3_2 - _1_2, _1);
            assert_eq!(_1 - _neg1_2, _3_2);
        }

        #[test]
        fn test_mul() {
            assert_eq!(_1 * _1_2, _1_2);
            assert_eq!(_1_2 * _3_2, Ratio::new(3,4));
            assert_eq!(_1_2 * _neg1_2, Ratio::new(-1, 4));
        }

        #[test]
        fn test_quot() {
            assert_eq!(_1 / _1_2, _2);
            assert_eq!(_3_2 / _1_2, _1 + _2);
            assert_eq!(_1 / _neg1_2, _neg1_2 + _neg1_2 + _neg1_2 + _neg1_2);
        }

        #[test]
        fn test_rem() {
            assert_eq!(_3_2 % _1, _1_2);
            assert_eq!(_2 % _neg1_2, _0);
            assert_eq!(_1_2 % _2,  _1_2);
        }

        #[test]
        fn test_neg() {
            assert_eq!(-_0, _0);
            assert_eq!(-_1_2, _neg1_2);
            assert_eq!(-(-_1), _1);
        }
        #[test]
        fn test_zero() {
            assert_eq!(_0 + _0, _0);
            assert_eq!(_0 * _0, _0);
            assert_eq!(_0 * _1, _0);
            assert_eq!(_0 / _neg1_2, _0);
            assert_eq!(_0 - _0, _0);
        }
        #[test]
        #[should_fail]
        fn test_quot_0() {
            let _a =  _1 / _0;
        }
    }

    #[test]
    fn test_round() {
        assert_eq!(_1_2.ceil(), _1);
        assert_eq!(_1_2.floor(), _0);
        assert_eq!(_1_2.round(num::RoundToZero), _0);
        assert_eq!(_1_2.round(num::RoundFromZero), _1);

        assert_eq!(_neg1_2.ceil(), _0);
        assert_eq!(_neg1_2.floor(), -_1);
        assert_eq!(_neg1_2.round(num::RoundToZero), _0);
        assert_eq!(_neg1_2.round(num::RoundFromZero), -_1);

        assert_eq!(_1.ceil(), _1);
        assert_eq!(_1.floor(), _1);
        assert_eq!(_1.round(num::RoundToZero), _1);
        assert_eq!(_1.round(num::RoundFromZero), _1);
    }

    #[test]
    fn test_fract() {
        assert_eq!(_1.fract(), _0);
        assert_eq!(_neg1_2.fract(), _neg1_2);
        assert_eq!(_1_2.fract(), _1_2);
        assert_eq!(_3_2.fract(), _1_2);
    }

    #[test]
    fn test_to_from_str() {
        fn test(r: Rational, s: ~str) {
            assert_eq!(FromStr::from_str(s), Some(r));
            assert_eq!(r.to_str(), s);
        }
        test(_1, ~"1/1");
        test(_0, ~"0/1");
        test(_1_2, ~"1/2");
        test(_3_2, ~"3/2");
        test(_2, ~"2/1");
        test(_neg1_2, ~"-1/2");
    }
    #[test]
    fn test_from_str_fail() {
        fn test(s: &str) {
            assert_eq!(FromStr::from_str::<Rational>(s), None);
        }

        for ["0 /1", "abc", "", "1/", "--1/2","3/2/1"].each |&s| {
            test(s);
        }
    }

    #[test]
    fn test_to_from_str_radix() {
        fn test(r: Rational, s: ~str, n: uint) {
            assert_eq!(FromStrRadix::from_str_radix(s, n), Some(r));
            assert_eq!(r.to_str_radix(n), s);
        }
        fn test3(r: Rational, s: ~str) { test(r, s, 3) }
        fn test16(r: Rational, s: ~str) { test(r, s, 16) }

        test3(_1, ~"1/1");
        test3(_0, ~"0/1");
        test3(_1_2, ~"1/2");
        test3(_3_2, ~"10/2");
        test3(_2, ~"2/1");
        test3(_neg1_2, ~"-1/2");
        test3(_neg1_2 / _2, ~"-1/11");

        test16(_1, ~"1/1");
        test16(_0, ~"0/1");
        test16(_1_2, ~"1/2");
        test16(_3_2, ~"3/2");
        test16(_2, ~"2/1");
        test16(_neg1_2, ~"-1/2");
        test16(_neg1_2 / _2, ~"-1/4");
        test16(Ratio::new(13,15), ~"d/f");
        test16(_1_2*_1_2*_1_2*_1_2, ~"1/10");
    }

    #[test]
    fn test_from_str_radix_fail() {
        fn test(s: &str) {
            assert_eq!(FromStrRadix::from_str_radix::<Rational>(s, 3), None);
        }

        for ["0 /1", "abc", "", "1/", "--1/2","3/2/1", "3/2"].each |&s| {
            test(s);
        }
    }
}
