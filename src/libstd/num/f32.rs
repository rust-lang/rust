// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for 32-bits floats (`f32` type)

#![stable(feature = "rust1", since = "1.0.0")]
#![allow(missing_docs)]
#![allow(unsigned_negation)]
#![doc(primitive = "f32")]

use prelude::v1::*;

use intrinsics;
use libc::c_int;
use num::{Float, FpCategory};
use num::strconv;
use num::strconv::ExponentFormat::{ExpNone, ExpDec};
use num::strconv::SignificantDigits::{DigAll, DigMax, DigExact};
use num::strconv::SignFormat::SignNeg;

use core::num;

pub use core::f32::{RADIX, MANTISSA_DIGITS, DIGITS, EPSILON, MIN_VALUE};
pub use core::f32::{MIN_POS_VALUE, MAX_VALUE, MIN_EXP, MAX_EXP, MIN_10_EXP};
pub use core::f32::{MAX_10_EXP, NAN, INFINITY, NEG_INFINITY};
pub use core::f32::{MIN, MIN_POSITIVE, MAX};
pub use core::f32::consts;

#[allow(dead_code)]
mod cmath {
    use libc::{c_float, c_int};

    #[link_name = "m"]
    extern {
        pub fn acosf(n: c_float) -> c_float;
        pub fn asinf(n: c_float) -> c_float;
        pub fn atanf(n: c_float) -> c_float;
        pub fn atan2f(a: c_float, b: c_float) -> c_float;
        pub fn cbrtf(n: c_float) -> c_float;
        pub fn coshf(n: c_float) -> c_float;
        pub fn erff(n: c_float) -> c_float;
        pub fn erfcf(n: c_float) -> c_float;
        pub fn expm1f(n: c_float) -> c_float;
        pub fn fdimf(a: c_float, b: c_float) -> c_float;
        pub fn frexpf(n: c_float, value: &mut c_int) -> c_float;
        pub fn fmaxf(a: c_float, b: c_float) -> c_float;
        pub fn fminf(a: c_float, b: c_float) -> c_float;
        pub fn fmodf(a: c_float, b: c_float) -> c_float;
        pub fn nextafterf(x: c_float, y: c_float) -> c_float;
        pub fn hypotf(x: c_float, y: c_float) -> c_float;
        pub fn ldexpf(x: c_float, n: c_int) -> c_float;
        pub fn logbf(n: c_float) -> c_float;
        pub fn log1pf(n: c_float) -> c_float;
        pub fn ilogbf(n: c_float) -> c_int;
        pub fn modff(n: c_float, iptr: &mut c_float) -> c_float;
        pub fn sinhf(n: c_float) -> c_float;
        pub fn tanf(n: c_float) -> c_float;
        pub fn tanhf(n: c_float) -> c_float;
        pub fn tgammaf(n: c_float) -> c_float;

        #[cfg(unix)]
        pub fn lgammaf_r(n: c_float, sign: &mut c_int) -> c_float;

        #[cfg(windows)]
        #[link_name="__lgammaf_r"]
        pub fn lgammaf_r(n: c_float, sign: &mut c_int) -> c_float;
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Float for f32 {
    #[inline]
    fn nan() -> f32 { num::Float::nan() }
    #[inline]
    fn infinity() -> f32 { num::Float::infinity() }
    #[inline]
    fn neg_infinity() -> f32 { num::Float::neg_infinity() }
    #[inline]
    fn zero() -> f32 { num::Float::zero() }
    #[inline]
    fn neg_zero() -> f32 { num::Float::neg_zero() }
    #[inline]
    fn one() -> f32 { num::Float::one() }

    #[allow(deprecated)]
    #[inline]
    fn mantissa_digits(unused_self: Option<f32>) -> uint {
        num::Float::mantissa_digits(unused_self)
    }
    #[allow(deprecated)]
    #[inline]
    fn digits(unused_self: Option<f32>) -> uint { num::Float::digits(unused_self) }
    #[allow(deprecated)]
    #[inline]
    fn epsilon() -> f32 { num::Float::epsilon() }
    #[allow(deprecated)]
    #[inline]
    fn min_exp(unused_self: Option<f32>) -> int { num::Float::min_exp(unused_self) }
    #[allow(deprecated)]
    #[inline]
    fn max_exp(unused_self: Option<f32>) -> int { num::Float::max_exp(unused_self) }
    #[allow(deprecated)]
    #[inline]
    fn min_10_exp(unused_self: Option<f32>) -> int { num::Float::min_10_exp(unused_self) }
    #[allow(deprecated)]
    #[inline]
    fn max_10_exp(unused_self: Option<f32>) -> int { num::Float::max_10_exp(unused_self) }
    #[allow(deprecated)]
    #[inline]
    fn min_value() -> f32 { num::Float::min_value() }
    #[allow(deprecated)]
    #[inline]
    fn min_pos_value(unused_self: Option<f32>) -> f32 { num::Float::min_pos_value(unused_self) }
    #[allow(deprecated)]
    #[inline]
    fn max_value() -> f32 { num::Float::max_value() }

    #[inline]
    fn is_nan(self) -> bool { num::Float::is_nan(self) }
    #[inline]
    fn is_infinite(self) -> bool { num::Float::is_infinite(self) }
    #[inline]
    fn is_finite(self) -> bool { num::Float::is_finite(self) }
    #[inline]
    fn is_normal(self) -> bool { num::Float::is_normal(self) }
    #[inline]
    fn classify(self) -> FpCategory { num::Float::classify(self) }

    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) { num::Float::integer_decode(self) }

    #[inline]
    fn floor(self) -> f32 { num::Float::floor(self) }
    #[inline]
    fn ceil(self) -> f32 { num::Float::ceil(self) }
    #[inline]
    fn round(self) -> f32 { num::Float::round(self) }
    #[inline]
    fn trunc(self) -> f32 { num::Float::trunc(self) }
    #[inline]
    fn fract(self) -> f32 { num::Float::fract(self) }

    #[inline]
    fn abs(self) -> f32 { num::Float::abs(self) }
    #[inline]
    fn signum(self) -> f32 { num::Float::signum(self) }
    #[inline]
    fn is_positive(self) -> bool { num::Float::is_positive(self) }
    #[inline]
    fn is_negative(self) -> bool { num::Float::is_negative(self) }

    #[inline]
    fn mul_add(self, a: f32, b: f32) -> f32 { num::Float::mul_add(self, a, b) }
    #[inline]
    fn recip(self) -> f32 { num::Float::recip(self) }

    #[inline]
    fn powi(self, n: i32) -> f32 { num::Float::powi(self, n) }
    #[inline]
    fn powf(self, n: f32) -> f32 { num::Float::powf(self, n) }

    #[inline]
    fn sqrt(self) -> f32 { num::Float::sqrt(self) }
    #[inline]
    fn rsqrt(self) -> f32 { num::Float::rsqrt(self) }

    #[inline]
    fn exp(self) -> f32 { num::Float::exp(self) }
    #[inline]
    fn exp2(self) -> f32 { num::Float::exp2(self) }
    #[inline]
    fn ln(self) -> f32 { num::Float::ln(self) }
    #[inline]
    fn log(self, base: f32) -> f32 { num::Float::log(self, base) }
    #[inline]
    fn log2(self) -> f32 { num::Float::log2(self) }
    #[inline]
    fn log10(self) -> f32 { num::Float::log10(self) }
    #[inline]
    fn to_degrees(self) -> f32 { num::Float::to_degrees(self) }
    #[inline]
    fn to_radians(self) -> f32 { num::Float::to_radians(self) }

    /// Constructs a floating point number by multiplying `x` by 2 raised to the
    /// power of `exp`
    #[inline]
    fn ldexp(self, exp: isize) -> f32 {
        unsafe { cmath::ldexpf(self, exp as c_int) }
    }

    /// Breaks the number into a normalized fraction and a base-2 exponent,
    /// satisfying:
    ///
    /// - `self = x * pow(2, exp)`
    /// - `0.5 <= abs(x) < 1.0`
    #[inline]
    fn frexp(self) -> (f32, int) {
        unsafe {
            let mut exp = 0;
            let x = cmath::frexpf(self, &mut exp);
            (x, exp as int)
        }
    }

    /// Returns the next representable floating-point value in the direction of
    /// `other`.
    #[inline]
    fn next_after(self, other: f32) -> f32 {
        unsafe { cmath::nextafterf(self, other) }
    }

    #[inline]
    fn max(self, other: f32) -> f32 {
        unsafe { cmath::fmaxf(self, other) }
    }

    #[inline]
    fn min(self, other: f32) -> f32 {
        unsafe { cmath::fminf(self, other) }
    }

    #[inline]
    fn abs_sub(self, other: f32) -> f32 {
        unsafe { cmath::fdimf(self, other) }
    }

    #[inline]
    fn cbrt(self) -> f32 {
        unsafe { cmath::cbrtf(self) }
    }

    #[inline]
    fn hypot(self, other: f32) -> f32 {
        unsafe { cmath::hypotf(self, other) }
    }

    #[inline]
    fn sin(self) -> f32 {
        unsafe { intrinsics::sinf32(self) }
    }

    #[inline]
    fn cos(self) -> f32 {
        unsafe { intrinsics::cosf32(self) }
    }

    #[inline]
    fn tan(self) -> f32 {
        unsafe { cmath::tanf(self) }
    }

    #[inline]
    fn asin(self) -> f32 {
        unsafe { cmath::asinf(self) }
    }

    #[inline]
    fn acos(self) -> f32 {
        unsafe { cmath::acosf(self) }
    }

    #[inline]
    fn atan(self) -> f32 {
        unsafe { cmath::atanf(self) }
    }

    #[inline]
    fn atan2(self, other: f32) -> f32 {
        unsafe { cmath::atan2f(self, other) }
    }

    /// Simultaneously computes the sine and cosine of the number
    #[inline]
    fn sin_cos(self) -> (f32, f32) {
        (self.sin(), self.cos())
    }

    /// Returns the exponential of the number, minus `1`, in a way that is
    /// accurate even if the number is close to zero
    #[inline]
    fn exp_m1(self) -> f32 {
        unsafe { cmath::expm1f(self) }
    }

    /// Returns the natural logarithm of the number plus `1` (`ln(1+n)`) more
    /// accurately than if the operations were performed separately
    #[inline]
    fn ln_1p(self) -> f32 {
        unsafe { cmath::log1pf(self) }
    }

    #[inline]
    fn sinh(self) -> f32 {
        unsafe { cmath::sinhf(self) }
    }

    #[inline]
    fn cosh(self) -> f32 {
        unsafe { cmath::coshf(self) }
    }

    #[inline]
    fn tanh(self) -> f32 {
        unsafe { cmath::tanhf(self) }
    }

    /// Inverse hyperbolic sine
    ///
    /// # Returns
    ///
    /// - on success, the inverse hyperbolic sine of `self` will be returned
    /// - `self` if `self` is `0.0`, `-0.0`, `INFINITY`, or `NEG_INFINITY`
    /// - `NAN` if `self` is `NAN`
    #[inline]
    fn asinh(self) -> f32 {
        match self {
            NEG_INFINITY => NEG_INFINITY,
            x => (x + ((x * x) + 1.0).sqrt()).ln(),
        }
    }

    /// Inverse hyperbolic cosine
    ///
    /// # Returns
    ///
    /// - on success, the inverse hyperbolic cosine of `self` will be returned
    /// - `INFINITY` if `self` is `INFINITY`
    /// - `NAN` if `self` is `NAN` or `self < 1.0` (including `NEG_INFINITY`)
    #[inline]
    fn acosh(self) -> f32 {
        match self {
            x if x < 1.0 => Float::nan(),
            x => (x + ((x * x) - 1.0).sqrt()).ln(),
        }
    }

    /// Inverse hyperbolic tangent
    ///
    /// # Returns
    ///
    /// - on success, the inverse hyperbolic tangent of `self` will be returned
    /// - `self` if `self` is `0.0` or `-0.0`
    /// - `INFINITY` if `self` is `1.0`
    /// - `NEG_INFINITY` if `self` is `-1.0`
    /// - `NAN` if the `self` is `NAN` or outside the domain of `-1.0 <= self <= 1.0`
    ///   (including `INFINITY` and `NEG_INFINITY`)
    #[inline]
    fn atanh(self) -> f32 {
        0.5 * ((2.0 * self) / (1.0 - self)).ln_1p()
    }
}

#[cfg(not(test))]
#[lang = "f32"]
#[stable(feature = "rust1", since = "1.0.0")]
impl f32 {
    // inlined methods from `num::Float`
    /// Returns the `NaN` value.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    ///
    /// let nan: f32 = Float::nan();
    ///
    /// assert!(nan.is_nan());
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    #[inline]
    pub fn nan() -> f32 { num::Float::nan() }

    /// Returns the infinite value.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    /// use std::f32;
    ///
    /// let infinity: f32 = Float::infinity();
    ///
    /// assert!(infinity.is_infinite());
    /// assert!(!infinity.is_finite());
    /// assert!(infinity > f32::MAX);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    #[inline]
    pub fn infinity() -> f32 { num::Float::infinity() }

    /// Returns the negative infinite value.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    /// use std::f32;
    ///
    /// let neg_infinity: f32 = Float::neg_infinity();
    ///
    /// assert!(neg_infinity.is_infinite());
    /// assert!(!neg_infinity.is_finite());
    /// assert!(neg_infinity < f32::MIN);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    #[inline]
    pub fn neg_infinity() -> f32 { num::Float::neg_infinity() }

    /// Returns `0.0`.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    ///
    /// let inf: f32 = Float::infinity();
    /// let zero: f32 = Float::zero();
    /// let neg_zero: f32 = Float::neg_zero();
    ///
    /// assert_eq!(zero, neg_zero);
    /// assert_eq!(7.0f32/inf, zero);
    /// assert_eq!(zero * 10.0, zero);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    #[inline]
    pub fn zero() -> f32 { num::Float::zero() }

    /// Returns `-0.0`.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    ///
    /// let inf: f32 = Float::infinity();
    /// let zero: f32 = Float::zero();
    /// let neg_zero: f32 = Float::neg_zero();
    ///
    /// assert_eq!(zero, neg_zero);
    /// assert_eq!(7.0f32/inf, zero);
    /// assert_eq!(zero * 10.0, zero);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    #[inline]
    pub fn neg_zero() -> f32 { num::Float::neg_zero() }

    /// Returns `1.0`.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    ///
    /// let one: f32 = Float::one();
    ///
    /// assert_eq!(one, 1.0f32);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    #[inline]
    pub fn one() -> f32 { num::Float::one() }

    // FIXME (#5527): These should be associated constants

    /// Deprecated: use `std::f32::MANTISSA_DIGITS` or `std::f64::MANTISSA_DIGITS`
    /// instead.
    #[unstable(feature = "std_misc")]
    #[deprecated(since = "1.0.0",
                 reason = "use `std::f32::MANTISSA_DIGITS` or \
                           `std::f64::MANTISSA_DIGITS` as appropriate")]
    #[allow(deprecated)]
    #[inline]
    pub fn mantissa_digits(unused_self: Option<f32>) -> uint {
        num::Float::mantissa_digits(unused_self)
    }

    /// Deprecated: use `std::f32::DIGITS` or `std::f64::DIGITS` instead.
    #[unstable(feature = "std_misc")]
    #[deprecated(since = "1.0.0",
                 reason = "use `std::f32::DIGITS` or `std::f64::DIGITS` as appropriate")]
    #[allow(deprecated)]
    #[inline]
    pub fn digits(unused_self: Option<f32>) -> uint { num::Float::digits(unused_self) }

    /// Deprecated: use `std::f32::EPSILON` or `std::f64::EPSILON` instead.
    #[unstable(feature = "std_misc")]
    #[deprecated(since = "1.0.0",
                 reason = "use `std::f32::EPSILON` or `std::f64::EPSILON` as appropriate")]
    #[allow(deprecated)]
    #[inline]
    pub fn epsilon() -> f32 { num::Float::epsilon() }

    /// Deprecated: use `std::f32::MIN_EXP` or `std::f64::MIN_EXP` instead.
    #[unstable(feature = "std_misc")]
    #[deprecated(since = "1.0.0",
                 reason = "use `std::f32::MIN_EXP` or `std::f64::MIN_EXP` as appropriate")]
    #[allow(deprecated)]
    #[inline]
    pub fn min_exp(unused_self: Option<f32>) -> int { num::Float::min_exp(unused_self) }

    /// Deprecated: use `std::f32::MAX_EXP` or `std::f64::MAX_EXP` instead.
    #[unstable(feature = "std_misc")]
    #[deprecated(since = "1.0.0",
                 reason = "use `std::f32::MAX_EXP` or `std::f64::MAX_EXP` as appropriate")]
    #[allow(deprecated)]
    #[inline]
    pub fn max_exp(unused_self: Option<f32>) -> int { num::Float::max_exp(unused_self) }

    /// Deprecated: use `std::f32::MIN_10_EXP` or `std::f64::MIN_10_EXP` instead.
    #[unstable(feature = "std_misc")]
    #[deprecated(since = "1.0.0",
                 reason = "use `std::f32::MIN_10_EXP` or `std::f64::MIN_10_EXP` as appropriate")]
    #[allow(deprecated)]
    #[inline]
    pub fn min_10_exp(unused_self: Option<f32>) -> int { num::Float::min_10_exp(unused_self) }

    /// Deprecated: use `std::f32::MAX_10_EXP` or `std::f64::MAX_10_EXP` instead.
    #[unstable(feature = "std_misc")]
    #[deprecated(since = "1.0.0",
                 reason = "use `std::f32::MAX_10_EXP` or `std::f64::MAX_10_EXP` as appropriate")]
    #[allow(deprecated)]
    #[inline]
    pub fn max_10_exp(unused_self: Option<f32>) -> int { num::Float::max_10_exp(unused_self) }

    /// Returns the smallest finite value that this type can represent.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let x: f64 = Float::min_value();
    ///
    /// assert_eq!(x, f64::MIN);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    #[inline]
    #[allow(deprecated)]
    pub fn min_value() -> f32 { num::Float::min_value() }

    /// Returns the smallest normalized positive number that this type can represent.
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    #[inline]
    #[allow(deprecated)]
    pub fn min_pos_value(unused_self: Option<f32>) -> f32 { num::Float::min_pos_value(unused_self) }

    /// Returns the largest finite value that this type can represent.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let x: f64 = Float::max_value();
    /// assert_eq!(x, f64::MAX);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    #[inline]
    #[allow(deprecated)]
    pub fn max_value() -> f32 { num::Float::max_value() }

    /// Returns `true` if this value is `NaN` and false otherwise.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let nan = f64::NAN;
    /// let f = 7.0;
    ///
    /// assert!(nan.is_nan());
    /// assert!(!f.is_nan());
    /// ```
    #[unstable(feature = "std_misc", reason = "position is undecided")]
    #[inline]
    pub fn is_nan(self) -> bool { num::Float::is_nan(self) }

    /// Returns `true` if this value is positive infinity or negative infinity and
    /// false otherwise.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    /// use std::f32;
    ///
    /// let f = 7.0f32;
    /// let inf: f32 = Float::infinity();
    /// let neg_inf: f32 = Float::neg_infinity();
    /// let nan: f32 = f32::NAN;
    ///
    /// assert!(!f.is_infinite());
    /// assert!(!nan.is_infinite());
    ///
    /// assert!(inf.is_infinite());
    /// assert!(neg_inf.is_infinite());
    /// ```
    #[unstable(feature = "std_misc", reason = "position is undecided")]
    #[inline]
    pub fn is_infinite(self) -> bool { num::Float::is_infinite(self) }

    /// Returns `true` if this number is neither infinite nor `NaN`.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    /// use std::f32;
    ///
    /// let f = 7.0f32;
    /// let inf: f32 = Float::infinity();
    /// let neg_inf: f32 = Float::neg_infinity();
    /// let nan: f32 = f32::NAN;
    ///
    /// assert!(f.is_finite());
    ///
    /// assert!(!nan.is_finite());
    /// assert!(!inf.is_finite());
    /// assert!(!neg_inf.is_finite());
    /// ```
    #[unstable(feature = "std_misc", reason = "position is undecided")]
    #[inline]
    pub fn is_finite(self) -> bool { num::Float::is_finite(self) }

    /// Returns `true` if the number is neither zero, infinite,
    /// [subnormal][subnormal], or `NaN`.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    /// use std::f32;
    ///
    /// let min = f32::MIN_POSITIVE; // 1.17549435e-38f32
    /// let max = f32::MAX;
    /// let lower_than_min = 1.0e-40_f32;
    /// let zero = 0.0f32;
    ///
    /// assert!(min.is_normal());
    /// assert!(max.is_normal());
    ///
    /// assert!(!zero.is_normal());
    /// assert!(!f32::NAN.is_normal());
    /// assert!(!f32::INFINITY.is_normal());
    /// // Values between `0` and `min` are Subnormal.
    /// assert!(!lower_than_min.is_normal());
    /// ```
    /// [subnormal]: http://en.wikipedia.org/wiki/Denormal_number
    #[unstable(feature = "std_misc", reason = "position is undecided")]
    #[inline]
    pub fn is_normal(self) -> bool { num::Float::is_normal(self) }

    /// Returns the floating point category of the number. If only one property
    /// is going to be tested, it is generally faster to use the specific
    /// predicate instead.
    ///
    /// ```
    /// # #![feature(core)]
    /// use std::num::{Float, FpCategory};
    /// use std::f32;
    ///
    /// let num = 12.4f32;
    /// let inf = f32::INFINITY;
    ///
    /// assert_eq!(num.classify(), FpCategory::Normal);
    /// assert_eq!(inf.classify(), FpCategory::Infinite);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn classify(self) -> FpCategory { num::Float::classify(self) }

    /// Returns the mantissa, base 2 exponent, and sign as integers, respectively.
    /// The original number can be recovered by `sign * mantissa * 2 ^ exponent`.
    /// The floating point encoding is documented in the [Reference][floating-point].
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    ///
    /// let num = 2.0f32;
    ///
    /// // (8388608, -22, 1)
    /// let (mantissa, exponent, sign) = num.integer_decode();
    /// let sign_f = sign as f32;
    /// let mantissa_f = mantissa as f32;
    /// let exponent_f = num.powf(exponent as f32);
    ///
    /// // 1 * 8388608 * 2^(-22) == 2
    /// let abs_difference = (sign_f * mantissa_f * exponent_f - num).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    /// [floating-point]: ../../../../../reference.html#machine-types
    #[unstable(feature = "std_misc", reason = "signature is undecided")]
    #[inline]
    pub fn integer_decode(self) -> (u64, i16, i8) { num::Float::integer_decode(self) }

    /// Returns the largest integer less than or equal to a number.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let f = 3.99;
    /// let g = 3.0;
    ///
    /// assert_eq!(f.floor(), 3.0);
    /// assert_eq!(g.floor(), 3.0);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn floor(self) -> f32 { num::Float::floor(self) }

    /// Returns the smallest integer greater than or equal to a number.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let f = 3.01;
    /// let g = 4.0;
    ///
    /// assert_eq!(f.ceil(), 4.0);
    /// assert_eq!(g.ceil(), 4.0);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn ceil(self) -> f32 { num::Float::ceil(self) }

    /// Returns the nearest integer to a number. Round half-way cases away from
    /// `0.0`.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let f = 3.3;
    /// let g = -3.3;
    ///
    /// assert_eq!(f.round(), 3.0);
    /// assert_eq!(g.round(), -3.0);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn round(self) -> f32 { num::Float::round(self) }

    /// Return the integer part of a number.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let f = 3.3;
    /// let g = -3.7;
    ///
    /// assert_eq!(f.trunc(), 3.0);
    /// assert_eq!(g.trunc(), -3.0);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn trunc(self) -> f32 { num::Float::trunc(self) }

    /// Returns the fractional part of a number.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 3.5;
    /// let y = -3.5;
    /// let abs_difference_x = (x.fract() - 0.5).abs();
    /// let abs_difference_y = (y.fract() - (-0.5)).abs();
    ///
    /// assert!(abs_difference_x < 1e-10);
    /// assert!(abs_difference_y < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn fract(self) -> f32 { num::Float::fract(self) }

    /// Computes the absolute value of `self`. Returns `Float::nan()` if the
    /// number is `Float::nan()`.
    ///
    /// ```
    /// # #![feature(core, std_misc)]
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let x = 3.5;
    /// let y = -3.5;
    ///
    /// let abs_difference_x = (x.abs() - x).abs();
    /// let abs_difference_y = (y.abs() - (-y)).abs();
    ///
    /// assert!(abs_difference_x < 1e-10);
    /// assert!(abs_difference_y < 1e-10);
    ///
    /// assert!(f64::NAN.abs().is_nan());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn abs(self) -> f32 { num::Float::abs(self) }

    /// Returns a number that represents the sign of `self`.
    ///
    /// - `1.0` if the number is positive, `+0.0` or `Float::infinity()`
    /// - `-1.0` if the number is negative, `-0.0` or `Float::neg_infinity()`
    /// - `Float::nan()` if the number is `Float::nan()`
    ///
    /// ```
    /// # #![feature(core, std_misc)]
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let f = 3.5;
    ///
    /// assert_eq!(f.signum(), 1.0);
    /// assert_eq!(f64::NEG_INFINITY.signum(), -1.0);
    ///
    /// assert!(f64::NAN.signum().is_nan());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn signum(self) -> f32 { num::Float::signum(self) }

    /// Returns `true` if `self` is positive, including `+0.0` and
    /// `Float::infinity()`.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let nan: f64 = f64::NAN;
    ///
    /// let f = 7.0;
    /// let g = -7.0;
    ///
    /// assert!(f.is_positive());
    /// assert!(!g.is_positive());
    /// // Requires both tests to determine if is `NaN`
    /// assert!(!nan.is_positive() && !nan.is_negative());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_positive(self) -> bool { num::Float::is_positive(self) }

    /// Returns `true` if `self` is negative, including `-0.0` and
    /// `Float::neg_infinity()`.
    ///
    /// ```
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let nan = f64::NAN;
    ///
    /// let f = 7.0;
    /// let g = -7.0;
    ///
    /// assert!(!f.is_negative());
    /// assert!(g.is_negative());
    /// // Requires both tests to determine if is `NaN`.
    /// assert!(!nan.is_positive() && !nan.is_negative());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_negative(self) -> bool { num::Float::is_negative(self) }

    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding
    /// error. This produces a more accurate result with better performance than
    /// a separate multiplication operation followed by an add.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    ///
    /// let m = 10.0;
    /// let x = 4.0;
    /// let b = 60.0;
    ///
    /// // 100.0
    /// let abs_difference = (m.mul_add(x, b) - (m*x + b)).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    #[inline]
    pub fn mul_add(self, a: f32, b: f32) -> f32 { num::Float::mul_add(self, a, b) }

    /// Take the reciprocal (inverse) of a number, `1/x`.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    ///
    /// let x = 2.0;
    /// let abs_difference = (x.recip() - (1.0/x)).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    #[inline]
    pub fn recip(self) -> f32 { num::Float::recip(self) }

    /// Raise a number to an integer power.
    ///
    /// Using this function is generally faster than using `powf`
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 2.0;
    /// let abs_difference = (x.powi(2) - x*x).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn powi(self, n: i32) -> f32 { num::Float::powi(self, n) }

    /// Raise a number to a floating point power.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 2.0;
    /// let abs_difference = (x.powf(2.0) - x*x).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn powf(self, n: f32) -> f32 { num::Float::powf(self, n) }

    /// Take the square root of a number.
    ///
    /// Returns NaN if `self` is a negative number.
    ///
    /// ```
    /// # #![feature(core, std_misc)]
    /// use std::num::Float;
    ///
    /// let positive = 4.0;
    /// let negative = -4.0;
    ///
    /// let abs_difference = (positive.sqrt() - 2.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// assert!(negative.sqrt().is_nan());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn sqrt(self) -> f32 { num::Float::sqrt(self) }


    /// Take the reciprocal (inverse) square root of a number, `1/sqrt(x)`.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    ///
    /// let f = 4.0;
    ///
    /// let abs_difference = (f.rsqrt() - 0.5).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    #[inline]
    pub fn rsqrt(self) -> f32 { num::Float::rsqrt(self) }

    /// Returns `e^(self)`, (the exponential function).
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let one = 1.0;
    /// // e^1
    /// let e = one.exp();
    ///
    /// // ln(e) - 1 == 0
    /// let abs_difference = (e.ln() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn exp(self) -> f32 { num::Float::exp(self) }

    /// Returns `2^(self)`.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let f = 2.0;
    ///
    /// // 2^2 - 4 == 0
    /// let abs_difference = (f.exp2() - 4.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn exp2(self) -> f32 { num::Float::exp2(self) }

    /// Returns the natural logarithm of the number.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let one = 1.0;
    /// // e^1
    /// let e = one.exp();
    ///
    /// // ln(e) - 1 == 0
    /// let abs_difference = (e.ln() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn ln(self) -> f32 { num::Float::ln(self) }

    /// Returns the logarithm of the number with respect to an arbitrary base.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let ten = 10.0;
    /// let two = 2.0;
    ///
    /// // log10(10) - 1 == 0
    /// let abs_difference_10 = (ten.log(10.0) - 1.0).abs();
    ///
    /// // log2(2) - 1 == 0
    /// let abs_difference_2 = (two.log(2.0) - 1.0).abs();
    ///
    /// assert!(abs_difference_10 < 1e-10);
    /// assert!(abs_difference_2 < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn log(self, base: f32) -> f32 { num::Float::log(self, base) }

    /// Returns the base 2 logarithm of the number.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let two = 2.0;
    ///
    /// // log2(2) - 1 == 0
    /// let abs_difference = (two.log2() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn log2(self) -> f32 { num::Float::log2(self) }

    /// Returns the base 10 logarithm of the number.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let ten = 10.0;
    ///
    /// // log10(10) - 1 == 0
    /// let abs_difference = (ten.log10() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn log10(self) -> f32 { num::Float::log10(self) }

    /// Convert radians to degrees.
    ///
    /// ```
    /// # #![feature(std_misc, core)]
    /// use std::num::Float;
    /// use std::f64::consts;
    ///
    /// let angle = consts::PI;
    ///
    /// let abs_difference = (angle.to_degrees() - 180.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc", reason = "desirability is unclear")]
    #[inline]
    pub fn to_degrees(self) -> f32 { num::Float::to_degrees(self) }

    /// Convert degrees to radians.
    ///
    /// ```
    /// # #![feature(std_misc, core)]
    /// use std::num::Float;
    /// use std::f64::consts;
    ///
    /// let angle = 180.0;
    ///
    /// let abs_difference = (angle.to_radians() - consts::PI).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc", reason = "desirability is unclear")]
    #[inline]
    pub fn to_radians(self) -> f32 { num::Float::to_radians(self) }

    /// Constructs a floating point number of `x*2^exp`.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    ///
    /// // 3*2^2 - 12 == 0
    /// let abs_difference = (Float::ldexp(3.0, 2) - 12.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "pending integer conventions")]
    #[inline]
    pub fn ldexp(x: f32, exp: int) -> f32 {
        unsafe { cmath::ldexpf(x, exp as c_int) }
    }

    /// Breaks the number into a normalized fraction and a base-2 exponent,
    /// satisfying:
    ///
    ///  * `self = x * 2^exp`
    ///  * `0.5 <= abs(x) < 1.0`
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    ///
    /// let x = 4.0;
    ///
    /// // (1/2)*2^3 -> 1 * 8/2 -> 4.0
    /// let f = x.frexp();
    /// let abs_difference_0 = (f.0 - 0.5).abs();
    /// let abs_difference_1 = (f.1 as f64 - 3.0).abs();
    ///
    /// assert!(abs_difference_0 < 1e-10);
    /// assert!(abs_difference_1 < 1e-10);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "pending integer conventions")]
    #[inline]
    pub fn frexp(self) -> (f32, int) {
        unsafe {
            let mut exp = 0;
            let x = cmath::frexpf(self, &mut exp);
            (x, exp as int)
        }
    }

    /// Returns the next representable floating-point value in the direction of
    /// `other`.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    ///
    /// let x = 1.0f32;
    ///
    /// let abs_diff = (x.next_after(2.0) - 1.00000011920928955078125_f32).abs();
    ///
    /// assert!(abs_diff < 1e-10);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    #[inline]
    pub fn next_after(self, other: f32) -> f32 {
        unsafe { cmath::nextafterf(self, other) }
    }

    /// Returns the maximum of the two numbers.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 1.0;
    /// let y = 2.0;
    ///
    /// assert_eq!(x.max(y), y);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn max(self, other: f32) -> f32 {
        unsafe { cmath::fmaxf(self, other) }
    }

    /// Returns the minimum of the two numbers.
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let x = 1.0;
    /// let y = 2.0;
    ///
    /// assert_eq!(x.min(y), x);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn min(self, other: f32) -> f32 {
        unsafe { cmath::fminf(self, other) }
    }

    /// The positive difference of two numbers.
    ///
    /// * If `self <= other`: `0:0`
    /// * Else: `self - other`
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    ///
    /// let x = 3.0;
    /// let y = -3.0;
    ///
    /// let abs_difference_x = (x.abs_sub(1.0) - 2.0).abs();
    /// let abs_difference_y = (y.abs_sub(1.0) - 0.0).abs();
    ///
    /// assert!(abs_difference_x < 1e-10);
    /// assert!(abs_difference_y < 1e-10);
    /// ```
    #[unstable(feature = "std_misc", reason = "may be renamed")]
    #[inline]
    pub fn abs_sub(self, other: f32) -> f32 {
        unsafe { cmath::fdimf(self, other) }
    }

    /// Take the cubic root of a number.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    ///
    /// let x = 8.0;
    ///
    /// // x^(1/3) - 2 == 0
    /// let abs_difference = (x.cbrt() - 2.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc", reason = "may be renamed")]
    #[inline]
    pub fn cbrt(self) -> f32 {
        unsafe { cmath::cbrtf(self) }
    }

    /// Calculate the length of the hypotenuse of a right-angle triangle given
    /// legs of length `x` and `y`.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    ///
    /// let x = 2.0;
    /// let y = 3.0;
    ///
    /// // sqrt(x^2 + y^2)
    /// let abs_difference = (x.hypot(y) - (x.powi(2) + y.powi(2)).sqrt()).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc",
               reason = "unsure about its place in the world")]
    #[inline]
    pub fn hypot(self, other: f32) -> f32 {
        unsafe { cmath::hypotf(self, other) }
    }

    /// Computes the sine of a number (in radians).
    ///
    /// ```
    /// # #![feature(core)]
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let x = f64::consts::PI/2.0;
    ///
    /// let abs_difference = (x.sin() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn sin(self) -> f32 {
        unsafe { intrinsics::sinf32(self) }
    }

    /// Computes the cosine of a number (in radians).
    ///
    /// ```
    /// # #![feature(core)]
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let x = 2.0*f64::consts::PI;
    ///
    /// let abs_difference = (x.cos() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn cos(self) -> f32 {
        unsafe { intrinsics::cosf32(self) }
    }

    /// Computes the tangent of a number (in radians).
    ///
    /// ```
    /// # #![feature(core)]
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let x = f64::consts::PI/4.0;
    /// let abs_difference = (x.tan() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-14);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn tan(self) -> f32 {
        unsafe { cmath::tanf(self) }
    }

    /// Computes the arcsine of a number. Return value is in radians in
    /// the range [-pi/2, pi/2] or NaN if the number is outside the range
    /// [-1, 1].
    ///
    /// ```
    /// # #![feature(core)]
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let f = f64::consts::PI / 2.0;
    ///
    /// // asin(sin(pi/2))
    /// let abs_difference = (f.sin().asin() - f64::consts::PI / 2.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn asin(self) -> f32 {
        unsafe { cmath::asinf(self) }
    }

    /// Computes the arccosine of a number. Return value is in radians in
    /// the range [0, pi] or NaN if the number is outside the range
    /// [-1, 1].
    ///
    /// ```
    /// # #![feature(core)]
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let f = f64::consts::PI / 4.0;
    ///
    /// // acos(cos(pi/4))
    /// let abs_difference = (f.cos().acos() - f64::consts::PI / 4.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn acos(self) -> f32 {
        unsafe { cmath::acosf(self) }
    }

    /// Computes the arctangent of a number. Return value is in radians in the
    /// range [-pi/2, pi/2];
    ///
    /// ```
    /// use std::num::Float;
    ///
    /// let f = 1.0;
    ///
    /// // atan(tan(1))
    /// let abs_difference = (f.tan().atan() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn atan(self) -> f32 {
        unsafe { cmath::atanf(self) }
    }

    /// Computes the four quadrant arctangent of `self` (`y`) and `other` (`x`).
    ///
    /// * `x = 0`, `y = 0`: `0`
    /// * `x >= 0`: `arctan(y/x)` -> `[-pi/2, pi/2]`
    /// * `y >= 0`: `arctan(y/x) + pi` -> `(pi/2, pi]`
    /// * `y < 0`: `arctan(y/x) - pi` -> `(-pi, -pi/2)`
    ///
    /// ```
    /// # #![feature(core)]
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let pi = f64::consts::PI;
    /// // All angles from horizontal right (+x)
    /// // 45 deg counter-clockwise
    /// let x1 = 3.0;
    /// let y1 = -3.0;
    ///
    /// // 135 deg clockwise
    /// let x2 = -3.0;
    /// let y2 = 3.0;
    ///
    /// let abs_difference_1 = (y1.atan2(x1) - (-pi/4.0)).abs();
    /// let abs_difference_2 = (y2.atan2(x2) - 3.0*pi/4.0).abs();
    ///
    /// assert!(abs_difference_1 < 1e-10);
    /// assert!(abs_difference_2 < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn atan2(self, other: f32) -> f32 {
        unsafe { cmath::atan2f(self, other) }
    }

    /// Simultaneously computes the sine and cosine of the number, `x`. Returns
    /// `(sin(x), cos(x))`.
    ///
    /// ```
    /// # #![feature(core)]
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let x = f64::consts::PI/4.0;
    /// let f = x.sin_cos();
    ///
    /// let abs_difference_0 = (f.0 - x.sin()).abs();
    /// let abs_difference_1 = (f.1 - x.cos()).abs();
    ///
    /// assert!(abs_difference_0 < 1e-10);
    /// assert!(abs_difference_0 < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn sin_cos(self) -> (f32, f32) {
        (self.sin(), self.cos())
    }

    /// Returns `e^(self) - 1` in a way that is accurate even if the
    /// number is close to zero.
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::num::Float;
    ///
    /// let x = 7.0;
    ///
    /// // e^(ln(7)) - 1
    /// let abs_difference = (x.ln().exp_m1() - 6.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc", reason = "may be renamed")]
    #[inline]
    pub fn exp_m1(self) -> f32 {
        unsafe { cmath::expm1f(self) }
    }

    /// Returns `ln(1+n)` (natural logarithm) more accurately than if
    /// the operations were performed separately.
    ///
    /// ```
    /// # #![feature(std_misc, core)]
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let x = f64::consts::E - 1.0;
    ///
    /// // ln(1 + (e - 1)) == ln(e) == 1
    /// let abs_difference = (x.ln_1p() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[unstable(feature = "std_misc", reason = "may be renamed")]
    #[inline]
    pub fn ln_1p(self) -> f32 {
        unsafe { cmath::log1pf(self) }
    }

    /// Hyperbolic sine function.
    ///
    /// ```
    /// # #![feature(core)]
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let e = f64::consts::E;
    /// let x = 1.0;
    ///
    /// let f = x.sinh();
    /// // Solving sinh() at 1 gives `(e^2-1)/(2e)`
    /// let g = (e*e - 1.0)/(2.0*e);
    /// let abs_difference = (f - g).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn sinh(self) -> f32 {
        unsafe { cmath::sinhf(self) }
    }

    /// Hyperbolic cosine function.
    ///
    /// ```
    /// # #![feature(core)]
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let e = f64::consts::E;
    /// let x = 1.0;
    /// let f = x.cosh();
    /// // Solving cosh() at 1 gives this result
    /// let g = (e*e + 1.0)/(2.0*e);
    /// let abs_difference = (f - g).abs();
    ///
    /// // Same result
    /// assert!(abs_difference < 1.0e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn cosh(self) -> f32 {
        unsafe { cmath::coshf(self) }
    }

    /// Hyperbolic tangent function.
    ///
    /// ```
    /// # #![feature(core)]
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let e = f64::consts::E;
    /// let x = 1.0;
    ///
    /// let f = x.tanh();
    /// // Solving tanh() at 1 gives `(1 - e^(-2))/(1 + e^(-2))`
    /// let g = (1.0 - e.powi(-2))/(1.0 + e.powi(-2));
    /// let abs_difference = (f - g).abs();
    ///
    /// assert!(abs_difference < 1.0e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn tanh(self) -> f32 {
        unsafe { cmath::tanhf(self) }
    }

    /// Inverse hyperbolic sine function.
    ///
    /// ```
    /// # #![feature(core)]
    /// use std::num::Float;
    ///
    /// let x = 1.0;
    /// let f = x.sinh().asinh();
    ///
    /// let abs_difference = (f - x).abs();
    ///
    /// assert!(abs_difference < 1.0e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn asinh(self) -> f32 {
        match self {
            NEG_INFINITY => NEG_INFINITY,
            x => (x + ((x * x) + 1.0).sqrt()).ln(),
        }
    }

    /// Inverse hyperbolic cosine function.
    ///
    /// ```
    /// # #![feature(core)]
    /// use std::num::Float;
    ///
    /// let x = 1.0;
    /// let f = x.cosh().acosh();
    ///
    /// let abs_difference = (f - x).abs();
    ///
    /// assert!(abs_difference < 1.0e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn acosh(self) -> f32 {
        match self {
            x if x < 1.0 => Float::nan(),
            x => (x + ((x * x) - 1.0).sqrt()).ln(),
        }
    }

    /// Inverse hyperbolic tangent function.
    ///
    /// ```
    /// # #![feature(core)]
    /// use std::num::Float;
    /// use std::f64;
    ///
    /// let e = f64::consts::E;
    /// let f = e.tanh().atanh();
    ///
    /// let abs_difference = (f - e).abs();
    ///
    /// assert!(abs_difference < 1.0e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn atanh(self) -> f32 {
        0.5 * ((2.0 * self) / (1.0 - self)).ln_1p()
    }
}

//
// Section: String Conversions
//

/// Converts a float to a string
///
/// # Arguments
///
/// * num - The float value
#[inline]
#[unstable(feature = "std_misc", reason = "may be removed or relocated")]
pub fn to_string(num: f32) -> String {
    let (r, _) = strconv::float_to_str_common(
        num, 10, true, SignNeg, DigAll, ExpNone, false);
    r
}

/// Converts a float to a string in hexadecimal format
///
/// # Arguments
///
/// * num - The float value
#[inline]
#[unstable(feature = "std_misc", reason = "may be removed or relocated")]
pub fn to_str_hex(num: f32) -> String {
    let (r, _) = strconv::float_to_str_common(
        num, 16, true, SignNeg, DigAll, ExpNone, false);
    r
}

/// Converts a float to a string in a given radix, and a flag indicating
/// whether it's a special value
///
/// # Arguments
///
/// * num - The float value
/// * radix - The base to use
#[inline]
#[unstable(feature = "std_misc", reason = "may be removed or relocated")]
pub fn to_str_radix_special(num: f32, rdx: u32) -> (String, bool) {
    strconv::float_to_str_common(num, rdx, true, SignNeg, DigAll, ExpNone, false)
}

/// Converts a float to a string with exactly the number of
/// provided significant digits
///
/// # Arguments
///
/// * num - The float value
/// * digits - The number of significant digits
#[inline]
#[unstable(feature = "std_misc", reason = "may be removed or relocated")]
pub fn to_str_exact(num: f32, dig: uint) -> String {
    let (r, _) = strconv::float_to_str_common(
        num, 10, true, SignNeg, DigExact(dig), ExpNone, false);
    r
}

/// Converts a float to a string with a maximum number of
/// significant digits
///
/// # Arguments
///
/// * num - The float value
/// * digits - The number of significant digits
#[inline]
#[unstable(feature = "std_misc", reason = "may be removed or relocated")]
pub fn to_str_digits(num: f32, dig: uint) -> String {
    let (r, _) = strconv::float_to_str_common(
        num, 10, true, SignNeg, DigMax(dig), ExpNone, false);
    r
}

/// Converts a float to a string using the exponential notation with exactly the number of
/// provided digits after the decimal point in the significand
///
/// # Arguments
///
/// * num - The float value
/// * digits - The number of digits after the decimal point
/// * upper - Use `E` instead of `e` for the exponent sign
#[inline]
#[unstable(feature = "std_misc", reason = "may be removed or relocated")]
pub fn to_str_exp_exact(num: f32, dig: uint, upper: bool) -> String {
    let (r, _) = strconv::float_to_str_common(
        num, 10, true, SignNeg, DigExact(dig), ExpDec, upper);
    r
}

/// Converts a float to a string using the exponential notation with the maximum number of
/// digits after the decimal point in the significand
///
/// # Arguments
///
/// * num - The float value
/// * digits - The number of digits after the decimal point
/// * upper - Use `E` instead of `e` for the exponent sign
#[inline]
#[unstable(feature = "std_misc", reason = "may be removed or relocated")]
pub fn to_str_exp_digits(num: f32, dig: uint, upper: bool) -> String {
    let (r, _) = strconv::float_to_str_common(
        num, 10, true, SignNeg, DigMax(dig), ExpDec, upper);
    r
}

#[cfg(test)]
mod tests {
    use f32::*;
    use num::*;
    use num::FpCategory as Fp;

    #[test]
    fn test_num_f32() {
        test_num(10f32, 2f32);
    }

    #[test]
    fn test_min_nan() {
        assert_eq!(NAN.min(2.0), 2.0);
        assert_eq!(2.0f32.min(NAN), 2.0);
    }

    #[test]
    fn test_max_nan() {
        assert_eq!(NAN.max(2.0), 2.0);
        assert_eq!(2.0f32.max(NAN), 2.0);
    }

    #[test]
    fn test_nan() {
        let nan: f32 = Float::nan();
        assert!(nan.is_nan());
        assert!(!nan.is_infinite());
        assert!(!nan.is_finite());
        assert!(!nan.is_normal());
        assert!(!nan.is_positive());
        assert!(!nan.is_negative());
        assert_eq!(Fp::Nan, nan.classify());
    }

    #[test]
    fn test_infinity() {
        let inf: f32 = Float::infinity();
        assert!(inf.is_infinite());
        assert!(!inf.is_finite());
        assert!(inf.is_positive());
        assert!(!inf.is_negative());
        assert!(!inf.is_nan());
        assert!(!inf.is_normal());
        assert_eq!(Fp::Infinite, inf.classify());
    }

    #[test]
    fn test_neg_infinity() {
        let neg_inf: f32 = Float::neg_infinity();
        assert!(neg_inf.is_infinite());
        assert!(!neg_inf.is_finite());
        assert!(!neg_inf.is_positive());
        assert!(neg_inf.is_negative());
        assert!(!neg_inf.is_nan());
        assert!(!neg_inf.is_normal());
        assert_eq!(Fp::Infinite, neg_inf.classify());
    }

    #[test]
    fn test_zero() {
        let zero: f32 = Float::zero();
        assert_eq!(0.0, zero);
        assert!(!zero.is_infinite());
        assert!(zero.is_finite());
        assert!(zero.is_positive());
        assert!(!zero.is_negative());
        assert!(!zero.is_nan());
        assert!(!zero.is_normal());
        assert_eq!(Fp::Zero, zero.classify());
    }

    #[test]
    fn test_neg_zero() {
        let neg_zero: f32 = Float::neg_zero();
        assert_eq!(0.0, neg_zero);
        assert!(!neg_zero.is_infinite());
        assert!(neg_zero.is_finite());
        assert!(!neg_zero.is_positive());
        assert!(neg_zero.is_negative());
        assert!(!neg_zero.is_nan());
        assert!(!neg_zero.is_normal());
        assert_eq!(Fp::Zero, neg_zero.classify());
    }

    #[test]
    fn test_one() {
        let one: f32 = Float::one();
        assert_eq!(1.0, one);
        assert!(!one.is_infinite());
        assert!(one.is_finite());
        assert!(one.is_positive());
        assert!(!one.is_negative());
        assert!(!one.is_nan());
        assert!(one.is_normal());
        assert_eq!(Fp::Normal, one.classify());
    }

    #[test]
    fn test_is_nan() {
        let nan: f32 = Float::nan();
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        assert!(nan.is_nan());
        assert!(!0.0f32.is_nan());
        assert!(!5.3f32.is_nan());
        assert!(!(-10.732f32).is_nan());
        assert!(!inf.is_nan());
        assert!(!neg_inf.is_nan());
    }

    #[test]
    fn test_is_infinite() {
        let nan: f32 = Float::nan();
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        assert!(!nan.is_infinite());
        assert!(inf.is_infinite());
        assert!(neg_inf.is_infinite());
        assert!(!0.0f32.is_infinite());
        assert!(!42.8f32.is_infinite());
        assert!(!(-109.2f32).is_infinite());
    }

    #[test]
    fn test_is_finite() {
        let nan: f32 = Float::nan();
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        assert!(!nan.is_finite());
        assert!(!inf.is_finite());
        assert!(!neg_inf.is_finite());
        assert!(0.0f32.is_finite());
        assert!(42.8f32.is_finite());
        assert!((-109.2f32).is_finite());
    }

    #[test]
    fn test_is_normal() {
        let nan: f32 = Float::nan();
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        let zero: f32 = Float::zero();
        let neg_zero: f32 = Float::neg_zero();
        assert!(!nan.is_normal());
        assert!(!inf.is_normal());
        assert!(!neg_inf.is_normal());
        assert!(!zero.is_normal());
        assert!(!neg_zero.is_normal());
        assert!(1f32.is_normal());
        assert!(1e-37f32.is_normal());
        assert!(!1e-38f32.is_normal());
    }

    #[test]
    fn test_classify() {
        let nan: f32 = Float::nan();
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        let zero: f32 = Float::zero();
        let neg_zero: f32 = Float::neg_zero();
        assert_eq!(nan.classify(), Fp::Nan);
        assert_eq!(inf.classify(), Fp::Infinite);
        assert_eq!(neg_inf.classify(), Fp::Infinite);
        assert_eq!(zero.classify(), Fp::Zero);
        assert_eq!(neg_zero.classify(), Fp::Zero);
        assert_eq!(1f32.classify(), Fp::Normal);
        assert_eq!(1e-37f32.classify(), Fp::Normal);
        assert_eq!(1e-38f32.classify(), Fp::Subnormal);
    }

    #[test]
    fn test_integer_decode() {
        assert_eq!(3.14159265359f32.integer_decode(), (13176795, -22, 1));
        assert_eq!((-8573.5918555f32).integer_decode(), (8779358, -10, -1));
        assert_eq!(2f32.powf(100.0).integer_decode(), (8388608, 77, 1));
        assert_eq!(0f32.integer_decode(), (0, -150, 1));
        assert_eq!((-0f32).integer_decode(), (0, -150, -1));
        assert_eq!(INFINITY.integer_decode(), (8388608, 105, 1));
        assert_eq!(NEG_INFINITY.integer_decode(), (8388608, 105, -1));
        assert_eq!(NAN.integer_decode(), (12582912, 105, 1));
    }

    #[test]
    fn test_floor() {
        assert_approx_eq!(1.0f32.floor(), 1.0f32);
        assert_approx_eq!(1.3f32.floor(), 1.0f32);
        assert_approx_eq!(1.5f32.floor(), 1.0f32);
        assert_approx_eq!(1.7f32.floor(), 1.0f32);
        assert_approx_eq!(0.0f32.floor(), 0.0f32);
        assert_approx_eq!((-0.0f32).floor(), -0.0f32);
        assert_approx_eq!((-1.0f32).floor(), -1.0f32);
        assert_approx_eq!((-1.3f32).floor(), -2.0f32);
        assert_approx_eq!((-1.5f32).floor(), -2.0f32);
        assert_approx_eq!((-1.7f32).floor(), -2.0f32);
    }

    #[test]
    fn test_ceil() {
        assert_approx_eq!(1.0f32.ceil(), 1.0f32);
        assert_approx_eq!(1.3f32.ceil(), 2.0f32);
        assert_approx_eq!(1.5f32.ceil(), 2.0f32);
        assert_approx_eq!(1.7f32.ceil(), 2.0f32);
        assert_approx_eq!(0.0f32.ceil(), 0.0f32);
        assert_approx_eq!((-0.0f32).ceil(), -0.0f32);
        assert_approx_eq!((-1.0f32).ceil(), -1.0f32);
        assert_approx_eq!((-1.3f32).ceil(), -1.0f32);
        assert_approx_eq!((-1.5f32).ceil(), -1.0f32);
        assert_approx_eq!((-1.7f32).ceil(), -1.0f32);
    }

    #[test]
    fn test_round() {
        assert_approx_eq!(1.0f32.round(), 1.0f32);
        assert_approx_eq!(1.3f32.round(), 1.0f32);
        assert_approx_eq!(1.5f32.round(), 2.0f32);
        assert_approx_eq!(1.7f32.round(), 2.0f32);
        assert_approx_eq!(0.0f32.round(), 0.0f32);
        assert_approx_eq!((-0.0f32).round(), -0.0f32);
        assert_approx_eq!((-1.0f32).round(), -1.0f32);
        assert_approx_eq!((-1.3f32).round(), -1.0f32);
        assert_approx_eq!((-1.5f32).round(), -2.0f32);
        assert_approx_eq!((-1.7f32).round(), -2.0f32);
    }

    #[test]
    fn test_trunc() {
        assert_approx_eq!(1.0f32.trunc(), 1.0f32);
        assert_approx_eq!(1.3f32.trunc(), 1.0f32);
        assert_approx_eq!(1.5f32.trunc(), 1.0f32);
        assert_approx_eq!(1.7f32.trunc(), 1.0f32);
        assert_approx_eq!(0.0f32.trunc(), 0.0f32);
        assert_approx_eq!((-0.0f32).trunc(), -0.0f32);
        assert_approx_eq!((-1.0f32).trunc(), -1.0f32);
        assert_approx_eq!((-1.3f32).trunc(), -1.0f32);
        assert_approx_eq!((-1.5f32).trunc(), -1.0f32);
        assert_approx_eq!((-1.7f32).trunc(), -1.0f32);
    }

    #[test]
    fn test_fract() {
        assert_approx_eq!(1.0f32.fract(), 0.0f32);
        assert_approx_eq!(1.3f32.fract(), 0.3f32);
        assert_approx_eq!(1.5f32.fract(), 0.5f32);
        assert_approx_eq!(1.7f32.fract(), 0.7f32);
        assert_approx_eq!(0.0f32.fract(), 0.0f32);
        assert_approx_eq!((-0.0f32).fract(), -0.0f32);
        assert_approx_eq!((-1.0f32).fract(), -0.0f32);
        assert_approx_eq!((-1.3f32).fract(), -0.3f32);
        assert_approx_eq!((-1.5f32).fract(), -0.5f32);
        assert_approx_eq!((-1.7f32).fract(), -0.7f32);
    }

    #[test]
    fn test_abs() {
        assert_eq!(INFINITY.abs(), INFINITY);
        assert_eq!(1f32.abs(), 1f32);
        assert_eq!(0f32.abs(), 0f32);
        assert_eq!((-0f32).abs(), 0f32);
        assert_eq!((-1f32).abs(), 1f32);
        assert_eq!(NEG_INFINITY.abs(), INFINITY);
        assert_eq!((1f32/NEG_INFINITY).abs(), 0f32);
        assert!(NAN.abs().is_nan());
    }

    #[test]
    fn test_signum() {
        assert_eq!(INFINITY.signum(), 1f32);
        assert_eq!(1f32.signum(), 1f32);
        assert_eq!(0f32.signum(), 1f32);
        assert_eq!((-0f32).signum(), -1f32);
        assert_eq!((-1f32).signum(), -1f32);
        assert_eq!(NEG_INFINITY.signum(), -1f32);
        assert_eq!((1f32/NEG_INFINITY).signum(), -1f32);
        assert!(NAN.signum().is_nan());
    }

    #[test]
    fn test_is_positive() {
        assert!(INFINITY.is_positive());
        assert!(1f32.is_positive());
        assert!(0f32.is_positive());
        assert!(!(-0f32).is_positive());
        assert!(!(-1f32).is_positive());
        assert!(!NEG_INFINITY.is_positive());
        assert!(!(1f32/NEG_INFINITY).is_positive());
        assert!(!NAN.is_positive());
    }

    #[test]
    fn test_is_negative() {
        assert!(!INFINITY.is_negative());
        assert!(!1f32.is_negative());
        assert!(!0f32.is_negative());
        assert!((-0f32).is_negative());
        assert!((-1f32).is_negative());
        assert!(NEG_INFINITY.is_negative());
        assert!((1f32/NEG_INFINITY).is_negative());
        assert!(!NAN.is_negative());
    }

    #[test]
    fn test_mul_add() {
        let nan: f32 = Float::nan();
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        assert_approx_eq!(12.3f32.mul_add(4.5, 6.7), 62.05);
        assert_approx_eq!((-12.3f32).mul_add(-4.5, -6.7), 48.65);
        assert_approx_eq!(0.0f32.mul_add(8.9, 1.2), 1.2);
        assert_approx_eq!(3.4f32.mul_add(-0.0, 5.6), 5.6);
        assert!(nan.mul_add(7.8, 9.0).is_nan());
        assert_eq!(inf.mul_add(7.8, 9.0), inf);
        assert_eq!(neg_inf.mul_add(7.8, 9.0), neg_inf);
        assert_eq!(8.9f32.mul_add(inf, 3.2), inf);
        assert_eq!((-3.2f32).mul_add(2.4, neg_inf), neg_inf);
    }

    #[test]
    fn test_recip() {
        let nan: f32 = Float::nan();
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        assert_eq!(1.0f32.recip(), 1.0);
        assert_eq!(2.0f32.recip(), 0.5);
        assert_eq!((-0.4f32).recip(), -2.5);
        assert_eq!(0.0f32.recip(), inf);
        assert!(nan.recip().is_nan());
        assert_eq!(inf.recip(), 0.0);
        assert_eq!(neg_inf.recip(), 0.0);
    }

    #[test]
    fn test_powi() {
        let nan: f32 = Float::nan();
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        assert_eq!(1.0f32.powi(1), 1.0);
        assert_approx_eq!((-3.1f32).powi(2), 9.61);
        assert_approx_eq!(5.9f32.powi(-2), 0.028727);
        assert_eq!(8.3f32.powi(0), 1.0);
        assert!(nan.powi(2).is_nan());
        assert_eq!(inf.powi(3), inf);
        assert_eq!(neg_inf.powi(2), inf);
    }

    #[test]
    fn test_powf() {
        let nan: f32 = Float::nan();
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        assert_eq!(1.0f32.powf(1.0), 1.0);
        assert_approx_eq!(3.4f32.powf(4.5), 246.408218);
        assert_approx_eq!(2.7f32.powf(-3.2), 0.041652);
        assert_approx_eq!((-3.1f32).powf(2.0), 9.61);
        assert_approx_eq!(5.9f32.powf(-2.0), 0.028727);
        assert_eq!(8.3f32.powf(0.0), 1.0);
        assert!(nan.powf(2.0).is_nan());
        assert_eq!(inf.powf(2.0), inf);
        assert_eq!(neg_inf.powf(3.0), neg_inf);
    }

    #[test]
    fn test_sqrt_domain() {
        assert!(NAN.sqrt().is_nan());
        assert!(NEG_INFINITY.sqrt().is_nan());
        assert!((-1.0f32).sqrt().is_nan());
        assert_eq!((-0.0f32).sqrt(), -0.0);
        assert_eq!(0.0f32.sqrt(), 0.0);
        assert_eq!(1.0f32.sqrt(), 1.0);
        assert_eq!(INFINITY.sqrt(), INFINITY);
    }

    #[test]
    fn test_rsqrt() {
        let nan: f32 = Float::nan();
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        assert!(nan.rsqrt().is_nan());
        assert_eq!(inf.rsqrt(), 0.0);
        assert!(neg_inf.rsqrt().is_nan());
        assert!((-1.0f32).rsqrt().is_nan());
        assert_eq!((-0.0f32).rsqrt(), neg_inf);
        assert_eq!(0.0f32.rsqrt(), inf);
        assert_eq!(1.0f32.rsqrt(), 1.0);
        assert_eq!(4.0f32.rsqrt(), 0.5);
    }

    #[test]
    fn test_exp() {
        assert_eq!(1.0, 0.0f32.exp());
        assert_approx_eq!(2.718282, 1.0f32.exp());
        assert_approx_eq!(148.413162, 5.0f32.exp());

        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        let nan: f32 = Float::nan();
        assert_eq!(inf, inf.exp());
        assert_eq!(0.0, neg_inf.exp());
        assert!(nan.exp().is_nan());
    }

    #[test]
    fn test_exp2() {
        assert_eq!(32.0, 5.0f32.exp2());
        assert_eq!(1.0, 0.0f32.exp2());

        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        let nan: f32 = Float::nan();
        assert_eq!(inf, inf.exp2());
        assert_eq!(0.0, neg_inf.exp2());
        assert!(nan.exp2().is_nan());
    }

    #[test]
    fn test_ln() {
        let nan: f32 = Float::nan();
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        assert_approx_eq!(1.0f32.exp().ln(), 1.0);
        assert!(nan.ln().is_nan());
        assert_eq!(inf.ln(), inf);
        assert!(neg_inf.ln().is_nan());
        assert!((-2.3f32).ln().is_nan());
        assert_eq!((-0.0f32).ln(), neg_inf);
        assert_eq!(0.0f32.ln(), neg_inf);
        assert_approx_eq!(4.0f32.ln(), 1.386294);
    }

    #[test]
    fn test_log() {
        let nan: f32 = Float::nan();
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        assert_eq!(10.0f32.log(10.0), 1.0);
        assert_approx_eq!(2.3f32.log(3.5), 0.664858);
        assert_eq!(1.0f32.exp().log(1.0.exp()), 1.0);
        assert!(1.0f32.log(1.0).is_nan());
        assert!(1.0f32.log(-13.9).is_nan());
        assert!(nan.log(2.3).is_nan());
        assert_eq!(inf.log(10.0), inf);
        assert!(neg_inf.log(8.8).is_nan());
        assert!((-2.3f32).log(0.1).is_nan());
        assert_eq!((-0.0f32).log(2.0), neg_inf);
        assert_eq!(0.0f32.log(7.0), neg_inf);
    }

    #[test]
    fn test_log2() {
        let nan: f32 = Float::nan();
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        assert_approx_eq!(10.0f32.log2(), 3.321928);
        assert_approx_eq!(2.3f32.log2(), 1.201634);
        assert_approx_eq!(1.0f32.exp().log2(), 1.442695);
        assert!(nan.log2().is_nan());
        assert_eq!(inf.log2(), inf);
        assert!(neg_inf.log2().is_nan());
        assert!((-2.3f32).log2().is_nan());
        assert_eq!((-0.0f32).log2(), neg_inf);
        assert_eq!(0.0f32.log2(), neg_inf);
    }

    #[test]
    fn test_log10() {
        let nan: f32 = Float::nan();
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        assert_eq!(10.0f32.log10(), 1.0);
        assert_approx_eq!(2.3f32.log10(), 0.361728);
        assert_approx_eq!(1.0f32.exp().log10(), 0.434294);
        assert_eq!(1.0f32.log10(), 0.0);
        assert!(nan.log10().is_nan());
        assert_eq!(inf.log10(), inf);
        assert!(neg_inf.log10().is_nan());
        assert!((-2.3f32).log10().is_nan());
        assert_eq!((-0.0f32).log10(), neg_inf);
        assert_eq!(0.0f32.log10(), neg_inf);
    }

    #[test]
    fn test_to_degrees() {
        let pi: f32 = consts::PI;
        let nan: f32 = Float::nan();
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        assert_eq!(0.0f32.to_degrees(), 0.0);
        assert_approx_eq!((-5.8f32).to_degrees(), -332.315521);
        assert_eq!(pi.to_degrees(), 180.0);
        assert!(nan.to_degrees().is_nan());
        assert_eq!(inf.to_degrees(), inf);
        assert_eq!(neg_inf.to_degrees(), neg_inf);
    }

    #[test]
    fn test_to_radians() {
        let pi: f32 = consts::PI;
        let nan: f32 = Float::nan();
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        assert_eq!(0.0f32.to_radians(), 0.0);
        assert_approx_eq!(154.6f32.to_radians(), 2.698279);
        assert_approx_eq!((-332.31f32).to_radians(), -5.799903);
        assert_eq!(180.0f32.to_radians(), pi);
        assert!(nan.to_radians().is_nan());
        assert_eq!(inf.to_radians(), inf);
        assert_eq!(neg_inf.to_radians(), neg_inf);
    }

    #[test]
    fn test_ldexp() {
        // We have to use from_str until base-2 exponents
        // are supported in floating-point literals
        let f1: f32 = FromStrRadix::from_str_radix("1p-123", 16).unwrap();
        let f2: f32 = FromStrRadix::from_str_radix("1p-111", 16).unwrap();
        let f3: f32 = FromStrRadix::from_str_radix("1.Cp-12", 16).unwrap();
        assert_eq!(1f32.ldexp(-123), f1);
        assert_eq!(1f32.ldexp(-111), f2);
        assert_eq!(Float::ldexp(1.75f32, -12), f3);

        assert_eq!(Float::ldexp(0f32, -123), 0f32);
        assert_eq!(Float::ldexp(-0f32, -123), -0f32);

        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        let nan: f32 = Float::nan();
        assert_eq!(Float::ldexp(inf, -123), inf);
        assert_eq!(Float::ldexp(neg_inf, -123), neg_inf);
        assert!(Float::ldexp(nan, -123).is_nan());
    }

    #[test]
    fn test_frexp() {
        // We have to use from_str until base-2 exponents
        // are supported in floating-point literals
        let f1: f32 = FromStrRadix::from_str_radix("1p-123", 16).unwrap();
        let f2: f32 = FromStrRadix::from_str_radix("1p-111", 16).unwrap();
        let f3: f32 = FromStrRadix::from_str_radix("1.Cp-123", 16).unwrap();
        let (x1, exp1) = f1.frexp();
        let (x2, exp2) = f2.frexp();
        let (x3, exp3) = f3.frexp();
        assert_eq!((x1, exp1), (0.5f32, -122));
        assert_eq!((x2, exp2), (0.5f32, -110));
        assert_eq!((x3, exp3), (0.875f32, -122));
        assert_eq!(Float::ldexp(x1, exp1), f1);
        assert_eq!(Float::ldexp(x2, exp2), f2);
        assert_eq!(Float::ldexp(x3, exp3), f3);

        assert_eq!(0f32.frexp(), (0f32, 0));
        assert_eq!((-0f32).frexp(), (-0f32, 0));
    }

    #[test] #[cfg_attr(windows, ignore)] // FIXME #8755
    fn test_frexp_nowin() {
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        let nan: f32 = Float::nan();
        assert_eq!(match inf.frexp() { (x, _) => x }, inf);
        assert_eq!(match neg_inf.frexp() { (x, _) => x }, neg_inf);
        assert!(match nan.frexp() { (x, _) => x.is_nan() })
    }

    #[test]
    fn test_abs_sub() {
        assert_eq!((-1f32).abs_sub(1f32), 0f32);
        assert_eq!(1f32.abs_sub(1f32), 0f32);
        assert_eq!(1f32.abs_sub(0f32), 1f32);
        assert_eq!(1f32.abs_sub(-1f32), 2f32);
        assert_eq!(NEG_INFINITY.abs_sub(0f32), 0f32);
        assert_eq!(INFINITY.abs_sub(1f32), INFINITY);
        assert_eq!(0f32.abs_sub(NEG_INFINITY), INFINITY);
        assert_eq!(0f32.abs_sub(INFINITY), 0f32);
    }

    #[test]
    fn test_abs_sub_nowin() {
        assert!(NAN.abs_sub(-1f32).is_nan());
        assert!(1f32.abs_sub(NAN).is_nan());
    }

    #[test]
    fn test_asinh() {
        assert_eq!(0.0f32.asinh(), 0.0f32);
        assert_eq!((-0.0f32).asinh(), -0.0f32);

        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        let nan: f32 = Float::nan();
        assert_eq!(inf.asinh(), inf);
        assert_eq!(neg_inf.asinh(), neg_inf);
        assert!(nan.asinh().is_nan());
        assert_approx_eq!(2.0f32.asinh(), 1.443635475178810342493276740273105f32);
        assert_approx_eq!((-2.0f32).asinh(), -1.443635475178810342493276740273105f32);
    }

    #[test]
    fn test_acosh() {
        assert_eq!(1.0f32.acosh(), 0.0f32);
        assert!(0.999f32.acosh().is_nan());

        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        let nan: f32 = Float::nan();
        assert_eq!(inf.acosh(), inf);
        assert!(neg_inf.acosh().is_nan());
        assert!(nan.acosh().is_nan());
        assert_approx_eq!(2.0f32.acosh(), 1.31695789692481670862504634730796844f32);
        assert_approx_eq!(3.0f32.acosh(), 1.76274717403908605046521864995958461f32);
    }

    #[test]
    fn test_atanh() {
        assert_eq!(0.0f32.atanh(), 0.0f32);
        assert_eq!((-0.0f32).atanh(), -0.0f32);

        let inf32: f32 = Float::infinity();
        let neg_inf32: f32 = Float::neg_infinity();
        assert_eq!(1.0f32.atanh(), inf32);
        assert_eq!((-1.0f32).atanh(), neg_inf32);

        assert!(2f64.atanh().atanh().is_nan());
        assert!((-2f64).atanh().atanh().is_nan());

        let inf64: f32 = Float::infinity();
        let neg_inf64: f32 = Float::neg_infinity();
        let nan32: f32 = Float::nan();
        assert!(inf64.atanh().is_nan());
        assert!(neg_inf64.atanh().is_nan());
        assert!(nan32.atanh().is_nan());

        assert_approx_eq!(0.5f32.atanh(), 0.54930614433405484569762261846126285f32);
        assert_approx_eq!((-0.5f32).atanh(), -0.54930614433405484569762261846126285f32);
    }

    #[test]
    fn test_real_consts() {
        use super::consts;

        let pi: f32 = consts::PI;
        let two_pi: f32 = consts::PI_2;
        let frac_pi_2: f32 = consts::FRAC_PI_2;
        let frac_pi_3: f32 = consts::FRAC_PI_3;
        let frac_pi_4: f32 = consts::FRAC_PI_4;
        let frac_pi_6: f32 = consts::FRAC_PI_6;
        let frac_pi_8: f32 = consts::FRAC_PI_8;
        let frac_1_pi: f32 = consts::FRAC_1_PI;
        let frac_2_pi: f32 = consts::FRAC_2_PI;
        let frac_2_sqrtpi: f32 = consts::FRAC_2_SQRTPI;
        let sqrt2: f32 = consts::SQRT2;
        let frac_1_sqrt2: f32 = consts::FRAC_1_SQRT2;
        let e: f32 = consts::E;
        let log2_e: f32 = consts::LOG2_E;
        let log10_e: f32 = consts::LOG10_E;
        let ln_2: f32 = consts::LN_2;
        let ln_10: f32 = consts::LN_10;

        assert_approx_eq!(two_pi, 2f32 * pi);
        assert_approx_eq!(frac_pi_2, pi / 2f32);
        assert_approx_eq!(frac_pi_3, pi / 3f32);
        assert_approx_eq!(frac_pi_4, pi / 4f32);
        assert_approx_eq!(frac_pi_6, pi / 6f32);
        assert_approx_eq!(frac_pi_8, pi / 8f32);
        assert_approx_eq!(frac_1_pi, 1f32 / pi);
        assert_approx_eq!(frac_2_pi, 2f32 / pi);
        assert_approx_eq!(frac_2_sqrtpi, 2f32 / pi.sqrt());
        assert_approx_eq!(sqrt2, 2f32.sqrt());
        assert_approx_eq!(frac_1_sqrt2, 1f32 / 2f32.sqrt());
        assert_approx_eq!(log2_e, e.log2());
        assert_approx_eq!(log10_e, e.log10());
        assert_approx_eq!(ln_2, 2f32.ln());
        assert_approx_eq!(ln_10, 10f32.ln());
    }
}
