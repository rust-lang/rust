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


use std::cmp;
use std::from_str::FromStr;
use std::num::{Zero,One,ToStrRadix,FromStrRadix,Round};
use super::bigint::BigInt;

/// Represents the ratio between 2 numbers.
#[deriving(Clone)]
#[allow(missing_doc)]
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

impl<T: Clone + Integer + Ord>
    Ratio<T> {
    /// Create a ratio representing the integer `t`.
    #[inline]
    pub fn from_integer(t: T) -> Ratio<T> {
        Ratio::new_raw(t, One::one())
    }

    /// Create a ratio without checking for `denom == 0` or reducing.
    #[inline]
    pub fn new_raw(numer: T, denom: T) -> Ratio<T> {
        Ratio { numer: numer, denom: denom }
    }

    /// Create a new Ratio. Fails if `denom == 0`.
    #[inline]
    pub fn new(numer: T, denom: T) -> Ratio<T> {
        if denom == Zero::zero() {
            fail!("denominator == 0");
        }
        let mut ret = Ratio::new_raw(numer, denom);
        ret.reduce();
        ret
    }

    /// Put self into lowest terms, with denom > 0.
    fn reduce(&mut self) {
        let g : T = self.numer.gcd(&self.denom);

        // FIXME(#6050): overloaded operators force moves with generic types
        // self.numer /= g;
        self.numer = self.numer / g;
        // FIXME(#6050): overloaded operators force moves with generic types
        // self.denom /= g;
        self.denom = self.denom / g;

        // keep denom positive!
        if self.denom < Zero::zero() {
            self.numer = -self.numer;
            self.denom = -self.denom;
        }
    }

    /// Return a `reduce`d copy of self.
    fn reduced(&self) -> Ratio<T> {
        let mut ret = self.clone();
        ret.reduce();
        ret
    }
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

impl<T: Clone + Integer + Ord> Orderable for Ratio<T> {
    #[inline]
    fn min(&self, other: &Ratio<T>) -> Ratio<T> {
        if *self < *other { self.clone() } else { other.clone() }
    }

    #[inline]
    fn max(&self, other: &Ratio<T>) -> Ratio<T> {
        if *self > *other { self.clone() } else { other.clone() }
    }

    #[inline]
    fn clamp(&self, mn: &Ratio<T>, mx: &Ratio<T>) -> Ratio<T> {
        if *self > *mx { mx.clone()} else
        if *self < *mn { mn.clone() } else { self.clone() }
    }
}


/* Arithmetic */
// a/b * c/d = (a*c)/(b*d)
impl<T: Clone + Integer + Ord>
    Mul<Ratio<T>,Ratio<T>> for Ratio<T> {
    #[inline]
    fn mul(&self, rhs: &Ratio<T>) -> Ratio<T> {
        Ratio::new(self.numer * rhs.numer, self.denom * rhs.denom)
    }
}

// (a/b) / (c/d) = (a*d)/(b*c)
impl<T: Clone + Integer + Ord>
    Div<Ratio<T>,Ratio<T>> for Ratio<T> {
    #[inline]
    fn div(&self, rhs: &Ratio<T>) -> Ratio<T> {
        Ratio::new(self.numer * rhs.denom, self.denom * rhs.numer)
    }
}

// Abstracts the a/b `op` c/d = (a*d `op` b*d) / (b*d) pattern
macro_rules! arith_impl {
    (impl $imp:ident, $method:ident) => {
        impl<T: Clone + Integer + Ord>
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

impl<T: Clone + Integer + Ord>
    Neg<Ratio<T>> for Ratio<T> {
    #[inline]
    fn neg(&self) -> Ratio<T> {
        Ratio::new_raw(-self.numer, self.denom.clone())
    }
}

/* Constants */
impl<T: Clone + Integer + Ord>
    Zero for Ratio<T> {
    #[inline]
    fn zero() -> Ratio<T> {
        Ratio::new_raw(Zero::zero(), One::one())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Zero::zero()
    }
}

impl<T: Clone + Integer + Ord>
    One for Ratio<T> {
    #[inline]
    fn one() -> Ratio<T> {
        Ratio::new_raw(One::one(), One::one())
    }
}

impl<T: Clone + Integer + Ord>
    Num for Ratio<T> {}

/* Utils */
impl<T: Clone + Integer + Ord>
    Round for Ratio<T> {

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

    #[inline]
    fn round(&self) -> Ratio<T> {
        if *self < Zero::zero() {
            Ratio::from_integer((self.numer - self.denom + One::one()) / self.denom)
        } else {
            Ratio::from_integer((self.numer + self.denom - One::one()) / self.denom)
        }
    }

    #[inline]
    fn trunc(&self) -> Ratio<T> {
        Ratio::from_integer(self.numer / self.denom)
    }

    fn fract(&self) -> Ratio<T> {
        Ratio::new_raw(self.numer % self.denom, self.denom.clone())
    }
}

impl<T: Clone + Integer + Ord> Fractional for Ratio<T> {
    #[inline]
    fn recip(&self) -> Ratio<T> {
        Ratio::new_raw(self.denom.clone(), self.numer.clone())
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

impl<T: FromStr + Clone + Integer + Ord>
    FromStr for Ratio<T> {
    /// Parses `numer/denom`.
    fn from_str(s: &str) -> Option<Ratio<T>> {
        let split: ~[&str] = s.splitn_iter('/', 1).collect();
        if split.len() < 2 { return None; }
        do FromStr::from_str::<T>(split[0]).chain |a| {
            do FromStr::from_str::<T>(split[1]).chain |b| {
                Some(Ratio::new(a.clone(), b.clone()))
            }
        }
    }
}
impl<T: FromStrRadix + Clone + Integer + Ord>
    FromStrRadix for Ratio<T> {
    /// Parses `numer/denom` where the numbers are in base `radix`.
    fn from_str_radix(s: &str, radix: uint) -> Option<Ratio<T>> {
        let split: ~[&str] = s.splitn_iter('/', 1).collect();
        if split.len() < 2 { None }
        else {
            do FromStrRadix::from_str_radix::<T>(split[0], radix).chain |a| {
                do FromStrRadix::from_str_radix::<T>(split[1], radix).chain |b| {
                    Some(Ratio::new(a.clone(), b.clone()))
                }
            }
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use std::num::{Zero,One,FromStrRadix,IntConvertible};
    use std::from_str::FromStr;

    pub static _0 : Rational = Ratio { numer: 0, denom: 1};
    pub static _1 : Rational = Ratio { numer: 1, denom: 1};
    pub static _2: Rational = Ratio { numer: 2, denom: 1};
    pub static _1_2: Rational = Ratio { numer: 1, denom: 2};
    pub static _3_2: Rational = Ratio { numer: 3, denom: 2};
    pub static _neg1_2: Rational =  Ratio { numer: -1, denom: 2};

    pub fn to_big(n: Rational) -> BigRational {
        Ratio::new(
            IntConvertible::from_int(n.numer),
            IntConvertible::from_int(n.denom)
        )
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
            fn test(a: Rational, b: Rational, c: Rational) {
                assert_eq!(a + b, c);
                assert_eq!(to_big(a) + to_big(b), to_big(c));
            }

            test(_1, _1_2, _3_2);
            test(_1, _1, _2);
            test(_1_2, _3_2, _2);
            test(_1_2, _neg1_2, _0);
        }

        #[test]
        fn test_sub() {
            fn test(a: Rational, b: Rational, c: Rational) {
                assert_eq!(a - b, c);
                assert_eq!(to_big(a) - to_big(b), to_big(c))
            }

            test(_1, _1_2, _1_2);
            test(_3_2, _1_2, _1);
            test(_1, _neg1_2, _3_2);
        }

        #[test]
        fn test_mul() {
            fn test(a: Rational, b: Rational, c: Rational) {
                assert_eq!(a * b, c);
                assert_eq!(to_big(a) * to_big(b), to_big(c))
            }

            test(_1, _1_2, _1_2);
            test(_1_2, _3_2, Ratio::new(3,4));
            test(_1_2, _neg1_2, Ratio::new(-1, 4));
        }

        #[test]
        fn test_div() {
            fn test(a: Rational, b: Rational, c: Rational) {
                assert_eq!(a / b, c);
                assert_eq!(to_big(a) / to_big(b), to_big(c))
            }

            test(_1, _1_2, _2);
            test(_3_2, _1_2, _1 + _2);
            test(_1, _neg1_2, _neg1_2 + _neg1_2 + _neg1_2 + _neg1_2);
        }

        #[test]
        fn test_rem() {
            fn test(a: Rational, b: Rational, c: Rational) {
                assert_eq!(a % b, c);
                assert_eq!(to_big(a) % to_big(b), to_big(c))
            }

            test(_3_2, _1, _1_2);
            test(_2, _neg1_2, _0);
            test(_1_2, _2,  _1_2);
        }

        #[test]
        fn test_neg() {
            fn test(a: Rational, b: Rational) {
                assert_eq!(-a, b);
                assert_eq!(-to_big(a), to_big(b))
            }

            test(_0, _0);
            test(_1_2, _neg1_2);
            test(-_1, _1);
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
        fn test_div_0() {
            let _a =  _1 / _0;
        }
    }

    #[test]
    fn test_round() {
        assert_eq!(_1_2.ceil(), _1);
        assert_eq!(_1_2.floor(), _0);
        assert_eq!(_1_2.round(), _1);
        assert_eq!(_1_2.trunc(), _0);

        assert_eq!(_neg1_2.ceil(), _0);
        assert_eq!(_neg1_2.floor(), -_1);
        assert_eq!(_neg1_2.round(), -_1);
        assert_eq!(_neg1_2.trunc(), _0);

        assert_eq!(_1.ceil(), _1);
        assert_eq!(_1.floor(), _1);
        assert_eq!(_1.round(), _1);
        assert_eq!(_1.trunc(), _1);
    }

    #[test]
    fn test_fract() {
        assert_eq!(_1.fract(), _0);
        assert_eq!(_neg1_2.fract(), _neg1_2);
        assert_eq!(_1_2.fract(), _1_2);
        assert_eq!(_3_2.fract(), _1_2);
    }

    #[test]
    fn test_recip() {
        assert_eq!(_1 * _1.recip(), _1);
        assert_eq!(_2 * _2.recip(), _1);
        assert_eq!(_1_2 * _1_2.recip(), _1);
        assert_eq!(_3_2 * _3_2.recip(), _1);
        assert_eq!(_neg1_2 * _neg1_2.recip(), _1);
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

        let xs = ["0 /1", "abc", "", "1/", "--1/2","3/2/1"];
        for xs.iter().advance |&s| {
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

        let xs = ["0 /1", "abc", "", "1/", "--1/2","3/2/1", "3/2"];
        for xs.iter().advance |&s| {
            test(s);
        }
    }
}
