// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


//! Complex numbers.

use std::fmt;
use std::num::{Zero,One,ToStrRadix};

// FIXME #1284: handle complex NaN & infinity etc. This
// probably doesn't map to C's _Complex correctly.

/// A complex number in Cartesian form.
#[deriving(Eq,Clone)]
pub struct Complex<T> {
    /// Real portion of the complex number
    pub re: T,
    /// Imaginary portion of the complex number
    pub im: T
}

pub type Complex32 = Complex<f32>;
pub type Complex64 = Complex<f64>;

impl<T: Clone + Num> Complex<T> {
    /// Create a new Complex
    #[inline]
    pub fn new(re: T, im: T) -> Complex<T> {
        Complex { re: re, im: im }
    }

    /**
    Returns the square of the norm (since `T` doesn't necessarily
    have a sqrt function), i.e. `re^2 + im^2`.
    */
    #[inline]
    pub fn norm_sqr(&self) -> T {
        self.re * self.re + self.im * self.im
    }


    /// Returns the complex conjugate. i.e. `re - i im`
    #[inline]
    pub fn conj(&self) -> Complex<T> {
        Complex::new(self.re.clone(), -self.im)
    }


    /// Multiplies `self` by the scalar `t`.
    #[inline]
    pub fn scale(&self, t: T) -> Complex<T> {
        Complex::new(self.re * t, self.im * t)
    }

    /// Divides `self` by the scalar `t`.
    #[inline]
    pub fn unscale(&self, t: T) -> Complex<T> {
        Complex::new(self.re / t, self.im / t)
    }

    /// Returns `1/self`
    #[inline]
    pub fn inv(&self) -> Complex<T> {
        let norm_sqr = self.norm_sqr();
        Complex::new(self.re / norm_sqr,
                    -self.im / norm_sqr)
    }
}

impl<T: Clone + FloatMath> Complex<T> {
    /// Calculate |self|
    #[inline]
    pub fn norm(&self) -> T {
        self.re.hypot(self.im)
    }
}

impl<T: Clone + FloatMath> Complex<T> {
    /// Calculate the principal Arg of self.
    #[inline]
    pub fn arg(&self) -> T {
        self.im.atan2(self.re)
    }
    /// Convert to polar form (r, theta), such that `self = r * exp(i
    /// * theta)`
    #[inline]
    pub fn to_polar(&self) -> (T, T) {
        (self.norm(), self.arg())
    }
    /// Convert a polar representation into a complex number.
    #[inline]
    pub fn from_polar(r: &T, theta: &T) -> Complex<T> {
        Complex::new(*r * theta.cos(), *r * theta.sin())
    }
}

/* arithmetic */
// (a + i b) + (c + i d) == (a + c) + i (b + d)
impl<T: Clone + Num> Add<Complex<T>, Complex<T>> for Complex<T> {
    #[inline]
    fn add(&self, other: &Complex<T>) -> Complex<T> {
        Complex::new(self.re + other.re, self.im + other.im)
    }
}
// (a + i b) - (c + i d) == (a - c) + i (b - d)
impl<T: Clone + Num> Sub<Complex<T>, Complex<T>> for Complex<T> {
    #[inline]
    fn sub(&self, other: &Complex<T>) -> Complex<T> {
        Complex::new(self.re - other.re, self.im - other.im)
    }
}
// (a + i b) * (c + i d) == (a*c - b*d) + i (a*d + b*c)
impl<T: Clone + Num> Mul<Complex<T>, Complex<T>> for Complex<T> {
    #[inline]
    fn mul(&self, other: &Complex<T>) -> Complex<T> {
        Complex::new(self.re*other.re - self.im*other.im,
                   self.re*other.im + self.im*other.re)
    }
}

// (a + i b) / (c + i d) == [(a + i b) * (c - i d)] / (c*c + d*d)
//   == [(a*c + b*d) / (c*c + d*d)] + i [(b*c - a*d) / (c*c + d*d)]
impl<T: Clone + Num> Div<Complex<T>, Complex<T>> for Complex<T> {
    #[inline]
    fn div(&self, other: &Complex<T>) -> Complex<T> {
        let norm_sqr = other.norm_sqr();
        Complex::new((self.re*other.re + self.im*other.im) / norm_sqr,
                   (self.im*other.re - self.re*other.im) / norm_sqr)
    }
}

impl<T: Clone + Num> Neg<Complex<T>> for Complex<T> {
    #[inline]
    fn neg(&self) -> Complex<T> {
        Complex::new(-self.re, -self.im)
    }
}

/* constants */
impl<T: Clone + Num> Zero for Complex<T> {
    #[inline]
    fn zero() -> Complex<T> {
        Complex::new(Zero::zero(), Zero::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }
}

impl<T: Clone + Num> One for Complex<T> {
    #[inline]
    fn one() -> Complex<T> {
        Complex::new(One::one(), Zero::zero())
    }
}

/* string conversions */
impl<T: fmt::Show + Num + Ord> fmt::Show for Complex<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.im < Zero::zero() {
            write!(f, "{}-{}i", self.re, -self.im)
        } else {
            write!(f, "{}+{}i", self.re, self.im)
        }
    }
}

impl<T: ToStrRadix + Num + Ord> ToStrRadix for Complex<T> {
    fn to_str_radix(&self, radix: uint) -> String {
        if self.im < Zero::zero() {
            format_strbuf!("{}-{}i",
                           self.re.to_str_radix(radix),
                           (-self.im).to_str_radix(radix))
        } else {
            format_strbuf!("{}+{}i",
                           self.re.to_str_radix(radix),
                           self.im.to_str_radix(radix))
        }
    }
}

#[cfg(test)]
mod test {
    #![allow(non_uppercase_statics)]

    use super::{Complex64, Complex};
    use std::num::{Zero,One,Float};

    pub static _0_0i : Complex64 = Complex { re: 0.0, im: 0.0 };
    pub static _1_0i : Complex64 = Complex { re: 1.0, im: 0.0 };
    pub static _1_1i : Complex64 = Complex { re: 1.0, im: 1.0 };
    pub static _0_1i : Complex64 = Complex { re: 0.0, im: 1.0 };
    pub static _neg1_1i : Complex64 = Complex { re: -1.0, im: 1.0 };
    pub static _05_05i : Complex64 = Complex { re: 0.5, im: 0.5 };
    pub static all_consts : [Complex64, .. 5] = [_0_0i, _1_0i, _1_1i, _neg1_1i, _05_05i];

    #[test]
    fn test_consts() {
        // check our constants are what Complex::new creates
        fn test(c : Complex64, r : f64, i: f64) {
            assert_eq!(c, Complex::new(r,i));
        }
        test(_0_0i, 0.0, 0.0);
        test(_1_0i, 1.0, 0.0);
        test(_1_1i, 1.0, 1.0);
        test(_neg1_1i, -1.0, 1.0);
        test(_05_05i, 0.5, 0.5);

        assert_eq!(_0_0i, Zero::zero());
        assert_eq!(_1_0i, One::one());
    }

    #[test]
    #[ignore(cfg(target_arch = "x86"))]
    // FIXME #7158: (maybe?) currently failing on x86.
    fn test_norm() {
        fn test(c: Complex64, ns: f64) {
            assert_eq!(c.norm_sqr(), ns);
            assert_eq!(c.norm(), ns.sqrt())
        }
        test(_0_0i, 0.0);
        test(_1_0i, 1.0);
        test(_1_1i, 2.0);
        test(_neg1_1i, 2.0);
        test(_05_05i, 0.5);
    }

    #[test]
    fn test_scale_unscale() {
        assert_eq!(_05_05i.scale(2.0), _1_1i);
        assert_eq!(_1_1i.unscale(2.0), _05_05i);
        for &c in all_consts.iter() {
            assert_eq!(c.scale(2.0).unscale(2.0), c);
        }
    }

    #[test]
    fn test_conj() {
        for &c in all_consts.iter() {
            assert_eq!(c.conj(), Complex::new(c.re, -c.im));
            assert_eq!(c.conj().conj(), c);
        }
    }

    #[test]
    fn test_inv() {
        assert_eq!(_1_1i.inv(), _05_05i.conj());
        assert_eq!(_1_0i.inv(), _1_0i.inv());
    }

    #[test]
    #[should_fail]
    #[ignore]
    fn test_inv_zero() {
        // FIXME #5736: should this really fail, or just NaN?
        _0_0i.inv();
    }

    #[test]
    fn test_arg() {
        fn test(c: Complex64, arg: f64) {
            assert!((c.arg() - arg).abs() < 1.0e-6)
        }
        test(_1_0i, 0.0);
        test(_1_1i, 0.25 * Float::pi());
        test(_neg1_1i, 0.75 * Float::pi());
        test(_05_05i, 0.25 * Float::pi());
    }

    #[test]
    fn test_polar_conv() {
        fn test(c: Complex64) {
            let (r, theta) = c.to_polar();
            assert!((c - Complex::from_polar(&r, &theta)).norm() < 1e-6);
        }
        for &c in all_consts.iter() { test(c); }
    }

    mod arith {
        use super::{_0_0i, _1_0i, _1_1i, _0_1i, _neg1_1i, _05_05i, all_consts};
        use std::num::Zero;

        #[test]
        fn test_add() {
            assert_eq!(_05_05i + _05_05i, _1_1i);
            assert_eq!(_0_1i + _1_0i, _1_1i);
            assert_eq!(_1_0i + _neg1_1i, _0_1i);

            for &c in all_consts.iter() {
                assert_eq!(_0_0i + c, c);
                assert_eq!(c + _0_0i, c);
            }
        }

        #[test]
        fn test_sub() {
            assert_eq!(_05_05i - _05_05i, _0_0i);
            assert_eq!(_0_1i - _1_0i, _neg1_1i);
            assert_eq!(_0_1i - _neg1_1i, _1_0i);

            for &c in all_consts.iter() {
                assert_eq!(c - _0_0i, c);
                assert_eq!(c - c, _0_0i);
            }
        }

        #[test]
        fn test_mul() {
            assert_eq!(_05_05i * _05_05i, _0_1i.unscale(2.0));
            assert_eq!(_1_1i * _0_1i, _neg1_1i);

            // i^2 & i^4
            assert_eq!(_0_1i * _0_1i, -_1_0i);
            assert_eq!(_0_1i * _0_1i * _0_1i * _0_1i, _1_0i);

            for &c in all_consts.iter() {
                assert_eq!(c * _1_0i, c);
                assert_eq!(_1_0i * c, c);
            }
        }
        #[test]
        fn test_div() {
            assert_eq!(_neg1_1i / _0_1i, _1_1i);
            for &c in all_consts.iter() {
                if c != Zero::zero() {
                    assert_eq!(c / c, _1_0i);
                }
            }
        }
        #[test]
        fn test_neg() {
            assert_eq!(-_1_0i + _0_1i, _neg1_1i);
            assert_eq!((-_0_1i) * _0_1i, _1_0i);
            for &c in all_consts.iter() {
                assert_eq!(-(-c), c);
            }
        }
    }

    #[test]
    fn test_to_str() {
        fn test(c : Complex64, s: String) {
            assert_eq!(c.to_str().to_strbuf(), s);
        }
        test(_0_0i, "0+0i".to_strbuf());
        test(_1_0i, "1+0i".to_strbuf());
        test(_0_1i, "0+1i".to_strbuf());
        test(_1_1i, "1+1i".to_strbuf());
        test(_neg1_1i, "-1+1i".to_strbuf());
        test(-_neg1_1i, "1-1i".to_strbuf());
        test(_05_05i, "0.5+0.5i".to_strbuf());
    }
}
