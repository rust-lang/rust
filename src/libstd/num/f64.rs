// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for 64-bits floats (`f64` type)

#![allow(missing_doc)]

use prelude::*;

use from_str::FromStr;
use intrinsics;
use libc::{c_int};
use mem;
use num::{FPCategory, FPNaN, FPInfinite , FPZero, FPSubnormal, FPNormal};
use num::{strconv};
use num;

pub use core::f64::{RADIX, MANTISSA_DIGITS, DIGITS, EPSILON, MIN_VALUE};
pub use core::f64::{MIN_POS_VALUE, MAX_VALUE, MIN_EXP, MAX_EXP, MIN_10_EXP};
pub use core::f64::{MAX_10_EXP, NAN, INFINITY, NEG_INFINITY};
pub use core::f64::consts;

#[allow(dead_code)]
mod cmath {
    use libc::{c_double, c_int};

    #[link_name = "m"]
    extern {
        pub fn acos(n: c_double) -> c_double;
        pub fn asin(n: c_double) -> c_double;
        pub fn atan(n: c_double) -> c_double;
        pub fn atan2(a: c_double, b: c_double) -> c_double;
        pub fn cbrt(n: c_double) -> c_double;
        pub fn cosh(n: c_double) -> c_double;
        pub fn erf(n: c_double) -> c_double;
        pub fn erfc(n: c_double) -> c_double;
        pub fn expm1(n: c_double) -> c_double;
        pub fn fdim(a: c_double, b: c_double) -> c_double;
        pub fn fmax(a: c_double, b: c_double) -> c_double;
        pub fn fmin(a: c_double, b: c_double) -> c_double;
        pub fn fmod(a: c_double, b: c_double) -> c_double;
        pub fn nextafter(x: c_double, y: c_double) -> c_double;
        pub fn frexp(n: c_double, value: &mut c_int) -> c_double;
        pub fn hypot(x: c_double, y: c_double) -> c_double;
        pub fn ldexp(x: c_double, n: c_int) -> c_double;
        pub fn logb(n: c_double) -> c_double;
        pub fn log1p(n: c_double) -> c_double;
        pub fn ilogb(n: c_double) -> c_int;
        pub fn modf(n: c_double, iptr: &mut c_double) -> c_double;
        pub fn sinh(n: c_double) -> c_double;
        pub fn tan(n: c_double) -> c_double;
        pub fn tanh(n: c_double) -> c_double;
        pub fn tgamma(n: c_double) -> c_double;

        // These are commonly only available for doubles

        pub fn j0(n: c_double) -> c_double;
        pub fn j1(n: c_double) -> c_double;
        pub fn jn(i: c_int, n: c_double) -> c_double;

        pub fn y0(n: c_double) -> c_double;
        pub fn y1(n: c_double) -> c_double;
        pub fn yn(i: c_int, n: c_double) -> c_double;

        #[cfg(unix)]
        pub fn lgamma_r(n: c_double, sign: &mut c_int) -> c_double;
        #[cfg(windows)]
        #[link_name="__lgamma_r"]
        pub fn lgamma_r(n: c_double, sign: &mut c_int) -> c_double;
    }
}

impl Float for f64 {
    #[inline]
    fn nan() -> f64 { NAN }

    #[inline]
    fn infinity() -> f64 { INFINITY }

    #[inline]
    fn neg_infinity() -> f64 { NEG_INFINITY }

    #[inline]
    fn neg_zero() -> f64 { -0.0 }

    /// Returns `true` if the number is NaN
    #[inline]
    fn is_nan(self) -> bool { self != self }

    /// Returns `true` if the number is infinite
    #[inline]
    fn is_infinite(self) -> bool {
        self == Float::infinity() || self == Float::neg_infinity()
    }

    /// Returns `true` if the number is neither infinite or NaN
    #[inline]
    fn is_finite(self) -> bool {
        !(self.is_nan() || self.is_infinite())
    }

    /// Returns `true` if the number is neither zero, infinite, subnormal or NaN
    #[inline]
    fn is_normal(self) -> bool {
        self.classify() == FPNormal
    }

    /// Returns the floating point category of the number. If only one property
    /// is going to be tested, it is generally faster to use the specific
    /// predicate instead.
    fn classify(self) -> FPCategory {
        static EXP_MASK: u64 = 0x7ff0000000000000;
        static MAN_MASK: u64 = 0x000fffffffffffff;

        let bits: u64 = unsafe { mem::transmute(self) };
        match (bits & MAN_MASK, bits & EXP_MASK) {
            (0, 0)        => FPZero,
            (_, 0)        => FPSubnormal,
            (0, EXP_MASK) => FPInfinite,
            (_, EXP_MASK) => FPNaN,
            _             => FPNormal,
        }
    }

    #[inline]
    fn mantissa_digits(_: Option<f64>) -> uint { MANTISSA_DIGITS }

    #[inline]
    fn digits(_: Option<f64>) -> uint { DIGITS }

    #[inline]
    fn epsilon() -> f64 { EPSILON }

    #[inline]
    fn min_exp(_: Option<f64>) -> int { MIN_EXP }

    #[inline]
    fn max_exp(_: Option<f64>) -> int { MAX_EXP }

    #[inline]
    fn min_10_exp(_: Option<f64>) -> int { MIN_10_EXP }

    #[inline]
    fn max_10_exp(_: Option<f64>) -> int { MAX_10_EXP }

    #[inline]
    fn min_pos_value(_: Option<f64>) -> f64 { MIN_POS_VALUE }

    /// Constructs a floating point number by multiplying `x` by 2 raised to the
    /// power of `exp`
    #[inline]
    fn ldexp(x: f64, exp: int) -> f64 {
        unsafe { cmath::ldexp(x, exp as c_int) }
    }

    /// Breaks the number into a normalized fraction and a base-2 exponent,
    /// satisfying:
    ///
    /// - `self = x * pow(2, exp)`
    /// - `0.5 <= abs(x) < 1.0`
    #[inline]
    fn frexp(self) -> (f64, int) {
        unsafe {
            let mut exp = 0;
            let x = cmath::frexp(self, &mut exp);
            (x, exp as int)
        }
    }

    /// Returns the mantissa, exponent and sign as integers.
    fn integer_decode(self) -> (u64, i16, i8) {
        let bits: u64 = unsafe { mem::transmute(self) };
        let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
        let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
        let mantissa = if exponent == 0 {
            (bits & 0xfffffffffffff) << 1
        } else {
            (bits & 0xfffffffffffff) | 0x10000000000000
        };
        // Exponent bias + mantissa shift
        exponent -= 1023 + 52;
        (mantissa, exponent, sign)
    }

    /// Returns the next representable floating-point value in the direction of
    /// `other`.
    #[inline]
    fn next_after(self, other: f64) -> f64 {
        unsafe { cmath::nextafter(self, other) }
    }

    /// Round half-way cases toward `NEG_INFINITY`
    #[inline]
    fn floor(self) -> f64 {
        unsafe { intrinsics::floorf64(self) }
    }

    /// Round half-way cases toward `INFINITY`
    #[inline]
    fn ceil(self) -> f64 {
        unsafe { intrinsics::ceilf64(self) }
    }

    /// Round half-way cases away from `0.0`
    #[inline]
    fn round(self) -> f64 {
        unsafe { intrinsics::roundf64(self) }
    }

    /// The integer part of the number (rounds towards `0.0`)
    #[inline]
    fn trunc(self) -> f64 {
        unsafe { intrinsics::truncf64(self) }
    }

    /// The fractional part of the number, satisfying:
    ///
    /// ```rust
    /// let x = 1.65f64;
    /// assert!(x == x.trunc() + x.fract())
    /// ```
    #[inline]
    fn fract(self) -> f64 { self - self.trunc() }

    #[inline]
    fn max(self, other: f64) -> f64 {
        unsafe { cmath::fmax(self, other) }
    }

    #[inline]
    fn min(self, other: f64) -> f64 {
        unsafe { cmath::fmin(self, other) }
    }

    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding
    /// error. This produces a more accurate result with better performance than
    /// a separate multiplication operation followed by an add.
    #[inline]
    fn mul_add(self, a: f64, b: f64) -> f64 {
        unsafe { intrinsics::fmaf64(self, a, b) }
    }

    /// The reciprocal (multiplicative inverse) of the number
    #[inline]
    fn recip(self) -> f64 { 1.0 / self }

    #[inline]
    fn powf(self, n: f64) -> f64 {
        unsafe { intrinsics::powf64(self, n) }
    }

    #[inline]
    fn powi(self, n: i32) -> f64 {
        unsafe { intrinsics::powif64(self, n) }
    }

    /// sqrt(2.0)
    #[inline]
    fn sqrt2() -> f64 { consts::SQRT2 }

    /// 1.0 / sqrt(2.0)
    #[inline]
    fn frac_1_sqrt2() -> f64 { consts::FRAC_1_SQRT2 }

    #[inline]
    fn sqrt(self) -> f64 {
        unsafe { intrinsics::sqrtf64(self) }
    }

    #[inline]
    fn rsqrt(self) -> f64 { self.sqrt().recip() }

    #[inline]
    fn cbrt(self) -> f64 {
        unsafe { cmath::cbrt(self) }
    }

    #[inline]
    fn hypot(self, other: f64) -> f64 {
        unsafe { cmath::hypot(self, other) }
    }

    /// Archimedes' constant
    #[inline]
    fn pi() -> f64 { consts::PI }

    /// 2.0 * pi
    #[inline]
    fn two_pi() -> f64 { consts::PI_2 }

    /// pi / 2.0
    #[inline]
    fn frac_pi_2() -> f64 { consts::FRAC_PI_2 }

    /// pi / 3.0
    #[inline]
    fn frac_pi_3() -> f64 { consts::FRAC_PI_3 }

    /// pi / 4.0
    #[inline]
    fn frac_pi_4() -> f64 { consts::FRAC_PI_4 }

    /// pi / 6.0
    #[inline]
    fn frac_pi_6() -> f64 { consts::FRAC_PI_6 }

    /// pi / 8.0
    #[inline]
    fn frac_pi_8() -> f64 { consts::FRAC_PI_8 }

    /// 1.0 / pi
    #[inline]
    fn frac_1_pi() -> f64 { consts::FRAC_1_PI }

    /// 2.0 / pi
    #[inline]
    fn frac_2_pi() -> f64 { consts::FRAC_2_PI }

    /// 2.0 / sqrt(pi)
    #[inline]
    fn frac_2_sqrtpi() -> f64 { consts::FRAC_2_SQRTPI }

    #[inline]
    fn sin(self) -> f64 {
        unsafe { intrinsics::sinf64(self) }
    }

    #[inline]
    fn cos(self) -> f64 {
        unsafe { intrinsics::cosf64(self) }
    }

    #[inline]
    fn tan(self) -> f64 {
        unsafe { cmath::tan(self) }
    }

    #[inline]
    fn asin(self) -> f64 {
        unsafe { cmath::asin(self) }
    }

    #[inline]
    fn acos(self) -> f64 {
        unsafe { cmath::acos(self) }
    }

    #[inline]
    fn atan(self) -> f64 {
        unsafe { cmath::atan(self) }
    }

    #[inline]
    fn atan2(self, other: f64) -> f64 {
        unsafe { cmath::atan2(self, other) }
    }

    /// Simultaneously computes the sine and cosine of the number
    #[inline]
    fn sin_cos(self) -> (f64, f64) {
        (self.sin(), self.cos())
    }

    /// Euler's number
    #[inline]
    fn e() -> f64 { consts::E }

    /// log2(e)
    #[inline]
    fn log2_e() -> f64 { consts::LOG2_E }

    /// log10(e)
    #[inline]
    fn log10_e() -> f64 { consts::LOG10_E }

    /// ln(2.0)
    #[inline]
    fn ln_2() -> f64 { consts::LN_2 }

    /// ln(10.0)
    #[inline]
    fn ln_10() -> f64 { consts::LN_10 }

    /// Returns the exponential of the number
    #[inline]
    fn exp(self) -> f64 {
        unsafe { intrinsics::expf64(self) }
    }

    /// Returns 2 raised to the power of the number
    #[inline]
    fn exp2(self) -> f64 {
        unsafe { intrinsics::exp2f64(self) }
    }

    /// Returns the exponential of the number, minus `1`, in a way that is
    /// accurate even if the number is close to zero
    #[inline]
    fn exp_m1(self) -> f64 {
        unsafe { cmath::expm1(self) }
    }

    /// Returns the natural logarithm of the number
    #[inline]
    fn ln(self) -> f64 {
        unsafe { intrinsics::logf64(self) }
    }

    /// Returns the logarithm of the number with respect to an arbitrary base
    #[inline]
    fn log(self, base: f64) -> f64 { self.ln() / base.ln() }

    /// Returns the base 2 logarithm of the number
    #[inline]
    fn log2(self) -> f64 {
        unsafe { intrinsics::log2f64(self) }
    }

    /// Returns the base 10 logarithm of the number
    #[inline]
    fn log10(self) -> f64 {
        unsafe { intrinsics::log10f64(self) }
    }

    /// Returns the natural logarithm of the number plus `1` (`ln(1+n)`) more
    /// accurately than if the operations were performed separately
    #[inline]
    fn ln_1p(self) -> f64 {
        unsafe { cmath::log1p(self) }
    }

    #[inline]
    fn sinh(self) -> f64 {
        unsafe { cmath::sinh(self) }
    }

    #[inline]
    fn cosh(self) -> f64 {
        unsafe { cmath::cosh(self) }
    }

    #[inline]
    fn tanh(self) -> f64 {
        unsafe { cmath::tanh(self) }
    }

    /// Inverse hyperbolic sine
    ///
    /// # Returns
    ///
    /// - on success, the inverse hyperbolic sine of `self` will be returned
    /// - `self` if `self` is `0.0`, `-0.0`, `INFINITY`, or `NEG_INFINITY`
    /// - `NAN` if `self` is `NAN`
    #[inline]
    fn asinh(self) -> f64 {
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
    fn acosh(self) -> f64 {
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
    fn atanh(self) -> f64 {
        0.5 * ((2.0 * self) / (1.0 - self)).ln_1p()
    }

    /// Converts to degrees, assuming the number is in radians
    #[inline]
    fn to_degrees(self) -> f64 { self * (180.0f64 / Float::pi()) }

    /// Converts to radians, assuming the number is in degrees
    #[inline]
    fn to_radians(self) -> f64 {
        let value: f64 = Float::pi();
        self * (value / 180.0)
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
pub fn to_str(num: f64) -> ~str {
    let (r, _) = strconv::float_to_str_common(
        num, 10u, true, strconv::SignNeg, strconv::DigAll, strconv::ExpNone, false);
    r
}

/// Converts a float to a string in hexadecimal format
///
/// # Arguments
///
/// * num - The float value
#[inline]
pub fn to_str_hex(num: f64) -> ~str {
    let (r, _) = strconv::float_to_str_common(
        num, 16u, true, strconv::SignNeg, strconv::DigAll, strconv::ExpNone, false);
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
pub fn to_str_radix_special(num: f64, rdx: uint) -> (~str, bool) {
    strconv::float_to_str_common(num, rdx, true,
                           strconv::SignNeg, strconv::DigAll, strconv::ExpNone, false)
}

/// Converts a float to a string with exactly the number of
/// provided significant digits
///
/// # Arguments
///
/// * num - The float value
/// * digits - The number of significant digits
#[inline]
pub fn to_str_exact(num: f64, dig: uint) -> ~str {
    let (r, _) = strconv::float_to_str_common(
        num, 10u, true, strconv::SignNeg, strconv::DigExact(dig), strconv::ExpNone, false);
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
pub fn to_str_digits(num: f64, dig: uint) -> ~str {
    let (r, _) = strconv::float_to_str_common(
        num, 10u, true, strconv::SignNeg, strconv::DigMax(dig), strconv::ExpNone, false);
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
pub fn to_str_exp_exact(num: f64, dig: uint, upper: bool) -> ~str {
    let (r, _) = strconv::float_to_str_common(
        num, 10u, true, strconv::SignNeg, strconv::DigExact(dig), strconv::ExpDec, upper);
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
pub fn to_str_exp_digits(num: f64, dig: uint, upper: bool) -> ~str {
    let (r, _) = strconv::float_to_str_common(
        num, 10u, true, strconv::SignNeg, strconv::DigMax(dig), strconv::ExpDec, upper);
    r
}

impl num::ToStrRadix for f64 {
    /// Converts a float to a string in a given radix
    ///
    /// # Arguments
    ///
    /// * num - The float value
    /// * radix - The base to use
    ///
    /// # Failure
    ///
    /// Fails if called on a special value like `inf`, `-inf` or `NAN` due to
    /// possible misinterpretation of the result at higher bases. If those values
    /// are expected, use `to_str_radix_special()` instead.
    #[inline]
    fn to_str_radix(&self, rdx: uint) -> ~str {
        let (r, special) = strconv::float_to_str_common(
            *self, rdx, true, strconv::SignNeg, strconv::DigAll, strconv::ExpNone, false);
        if special { fail!("number has a special value, \
                             try to_str_radix_special() if those are expected") }
        r
    }
}

/// Convert a string in base 16 to a float.
/// Accepts an optional binary exponent.
///
/// This function accepts strings such as
///
/// * 'a4.fe'
/// * '+a4.fe', equivalent to 'a4.fe'
/// * '-a4.fe'
/// * '2b.aP128', or equivalently, '2b.ap128'
/// * '2b.aP-128'
/// * '.' (understood as 0)
/// * 'c.'
/// * '.c', or, equivalently,  '0.c'
/// * '+inf', 'inf', '-inf', 'NaN'
///
/// Leading and trailing whitespace represent an error.
///
/// # Arguments
///
/// * num - A string
///
/// # Return value
///
/// `None` if the string did not represent a valid number.  Otherwise,
/// `Some(n)` where `n` is the floating-point number represented by `[num]`.
#[inline]
pub fn from_str_hex(num: &str) -> Option<f64> {
    strconv::from_str_common(num, 16u, true, true, true,
                             strconv::ExpBin, false, false)
}

impl FromStr for f64 {
    /// Convert a string in base 10 to a float.
    /// Accepts an optional decimal exponent.
    ///
    /// This function accepts strings such as
    ///
    /// * '3.14'
    /// * '+3.14', equivalent to '3.14'
    /// * '-3.14'
    /// * '2.5E10', or equivalently, '2.5e10'
    /// * '2.5E-10'
    /// * '.' (understood as 0)
    /// * '5.'
    /// * '.5', or, equivalently,  '0.5'
    /// * '+inf', 'inf', '-inf', 'NaN'
    ///
    /// Leading and trailing whitespace represent an error.
    ///
    /// # Arguments
    ///
    /// * num - A string
    ///
    /// # Return value
    ///
    /// `none` if the string did not represent a valid number.  Otherwise,
    /// `Some(n)` where `n` is the floating-point number represented by `num`.
    #[inline]
    fn from_str(val: &str) -> Option<f64> {
        strconv::from_str_common(val, 10u, true, true, true,
                                 strconv::ExpDec, false, false)
    }
}

impl num::FromStrRadix for f64 {
    /// Convert a string in a given base to a float.
    ///
    /// Due to possible conflicts, this function does **not** accept
    /// the special values `inf`, `-inf`, `+inf` and `NaN`, **nor**
    /// does it recognize exponents of any kind.
    ///
    /// Leading and trailing whitespace represent an error.
    ///
    /// # Arguments
    ///
    /// * num - A string
    /// * radix - The base to use. Must lie in the range [2 .. 36]
    ///
    /// # Return value
    ///
    /// `None` if the string did not represent a valid number. Otherwise,
    /// `Some(n)` where `n` is the floating-point number represented by `num`.
    #[inline]
    fn from_str_radix(val: &str, rdx: uint) -> Option<f64> {
        strconv::from_str_common(val, rdx, true, true, false,
                                 strconv::ExpNone, false, false)
    }
}

#[cfg(test)]
mod tests {
    use f64::*;
    use num::*;
    use num;

    #[test]
    fn test_min_nan() {
        assert_eq!(NAN.min(2.0), 2.0);
        assert_eq!(2.0f64.min(NAN), 2.0);
    }

    #[test]
    fn test_max_nan() {
        assert_eq!(NAN.max(2.0), 2.0);
        assert_eq!(2.0f64.max(NAN), 2.0);
    }

    #[test]
    fn test_num() {
        num::test_num(10f64, 2f64);
    }

    #[test]
    fn test_floor() {
        assert_approx_eq!(1.0f64.floor(), 1.0f64);
        assert_approx_eq!(1.3f64.floor(), 1.0f64);
        assert_approx_eq!(1.5f64.floor(), 1.0f64);
        assert_approx_eq!(1.7f64.floor(), 1.0f64);
        assert_approx_eq!(0.0f64.floor(), 0.0f64);
        assert_approx_eq!((-0.0f64).floor(), -0.0f64);
        assert_approx_eq!((-1.0f64).floor(), -1.0f64);
        assert_approx_eq!((-1.3f64).floor(), -2.0f64);
        assert_approx_eq!((-1.5f64).floor(), -2.0f64);
        assert_approx_eq!((-1.7f64).floor(), -2.0f64);
    }

    #[test]
    fn test_ceil() {
        assert_approx_eq!(1.0f64.ceil(), 1.0f64);
        assert_approx_eq!(1.3f64.ceil(), 2.0f64);
        assert_approx_eq!(1.5f64.ceil(), 2.0f64);
        assert_approx_eq!(1.7f64.ceil(), 2.0f64);
        assert_approx_eq!(0.0f64.ceil(), 0.0f64);
        assert_approx_eq!((-0.0f64).ceil(), -0.0f64);
        assert_approx_eq!((-1.0f64).ceil(), -1.0f64);
        assert_approx_eq!((-1.3f64).ceil(), -1.0f64);
        assert_approx_eq!((-1.5f64).ceil(), -1.0f64);
        assert_approx_eq!((-1.7f64).ceil(), -1.0f64);
    }

    #[test]
    fn test_round() {
        assert_approx_eq!(1.0f64.round(), 1.0f64);
        assert_approx_eq!(1.3f64.round(), 1.0f64);
        assert_approx_eq!(1.5f64.round(), 2.0f64);
        assert_approx_eq!(1.7f64.round(), 2.0f64);
        assert_approx_eq!(0.0f64.round(), 0.0f64);
        assert_approx_eq!((-0.0f64).round(), -0.0f64);
        assert_approx_eq!((-1.0f64).round(), -1.0f64);
        assert_approx_eq!((-1.3f64).round(), -1.0f64);
        assert_approx_eq!((-1.5f64).round(), -2.0f64);
        assert_approx_eq!((-1.7f64).round(), -2.0f64);
    }

    #[test]
    fn test_trunc() {
        assert_approx_eq!(1.0f64.trunc(), 1.0f64);
        assert_approx_eq!(1.3f64.trunc(), 1.0f64);
        assert_approx_eq!(1.5f64.trunc(), 1.0f64);
        assert_approx_eq!(1.7f64.trunc(), 1.0f64);
        assert_approx_eq!(0.0f64.trunc(), 0.0f64);
        assert_approx_eq!((-0.0f64).trunc(), -0.0f64);
        assert_approx_eq!((-1.0f64).trunc(), -1.0f64);
        assert_approx_eq!((-1.3f64).trunc(), -1.0f64);
        assert_approx_eq!((-1.5f64).trunc(), -1.0f64);
        assert_approx_eq!((-1.7f64).trunc(), -1.0f64);
    }

    #[test]
    fn test_fract() {
        assert_approx_eq!(1.0f64.fract(), 0.0f64);
        assert_approx_eq!(1.3f64.fract(), 0.3f64);
        assert_approx_eq!(1.5f64.fract(), 0.5f64);
        assert_approx_eq!(1.7f64.fract(), 0.7f64);
        assert_approx_eq!(0.0f64.fract(), 0.0f64);
        assert_approx_eq!((-0.0f64).fract(), -0.0f64);
        assert_approx_eq!((-1.0f64).fract(), -0.0f64);
        assert_approx_eq!((-1.3f64).fract(), -0.3f64);
        assert_approx_eq!((-1.5f64).fract(), -0.5f64);
        assert_approx_eq!((-1.7f64).fract(), -0.7f64);
    }

    #[test]
    fn test_asinh() {
        assert_eq!(0.0f64.asinh(), 0.0f64);
        assert_eq!((-0.0f64).asinh(), -0.0f64);

        let inf: f64 = Float::infinity();
        let neg_inf: f64 = Float::neg_infinity();
        let nan: f64 = Float::nan();
        assert_eq!(inf.asinh(), inf);
        assert_eq!(neg_inf.asinh(), neg_inf);
        assert!(nan.asinh().is_nan());
        assert_approx_eq!(2.0f64.asinh(), 1.443635475178810342493276740273105f64);
        assert_approx_eq!((-2.0f64).asinh(), -1.443635475178810342493276740273105f64);
    }

    #[test]
    fn test_acosh() {
        assert_eq!(1.0f64.acosh(), 0.0f64);
        assert!(0.999f64.acosh().is_nan());

        let inf: f64 = Float::infinity();
        let neg_inf: f64 = Float::neg_infinity();
        let nan: f64 = Float::nan();
        assert_eq!(inf.acosh(), inf);
        assert!(neg_inf.acosh().is_nan());
        assert!(nan.acosh().is_nan());
        assert_approx_eq!(2.0f64.acosh(), 1.31695789692481670862504634730796844f64);
        assert_approx_eq!(3.0f64.acosh(), 1.76274717403908605046521864995958461f64);
    }

    #[test]
    fn test_atanh() {
        assert_eq!(0.0f64.atanh(), 0.0f64);
        assert_eq!((-0.0f64).atanh(), -0.0f64);

        let inf: f64 = Float::infinity();
        let neg_inf: f64 = Float::neg_infinity();
        let nan: f64 = Float::nan();
        assert_eq!(1.0f64.atanh(), inf);
        assert_eq!((-1.0f64).atanh(), neg_inf);
        assert!(2f64.atanh().atanh().is_nan());
        assert!((-2f64).atanh().atanh().is_nan());
        assert!(inf.atanh().is_nan());
        assert!(neg_inf.atanh().is_nan());
        assert!(nan.atanh().is_nan());
        assert_approx_eq!(0.5f64.atanh(), 0.54930614433405484569762261846126285f64);
        assert_approx_eq!((-0.5f64).atanh(), -0.54930614433405484569762261846126285f64);
    }

    #[test]
    fn test_real_consts() {
        let pi: f64 = Float::pi();
        let two_pi: f64 = Float::two_pi();
        let frac_pi_2: f64 = Float::frac_pi_2();
        let frac_pi_3: f64 = Float::frac_pi_3();
        let frac_pi_4: f64 = Float::frac_pi_4();
        let frac_pi_6: f64 = Float::frac_pi_6();
        let frac_pi_8: f64 = Float::frac_pi_8();
        let frac_1_pi: f64 = Float::frac_1_pi();
        let frac_2_pi: f64 = Float::frac_2_pi();
        let frac_2_sqrtpi: f64 = Float::frac_2_sqrtpi();
        let sqrt2: f64 = Float::sqrt2();
        let frac_1_sqrt2: f64 = Float::frac_1_sqrt2();
        let e: f64 = Float::e();
        let log2_e: f64 = Float::log2_e();
        let log10_e: f64 = Float::log10_e();
        let ln_2: f64 = Float::ln_2();
        let ln_10: f64 = Float::ln_10();

        assert_approx_eq!(two_pi, 2.0 * pi);
        assert_approx_eq!(frac_pi_2, pi / 2f64);
        assert_approx_eq!(frac_pi_3, pi / 3f64);
        assert_approx_eq!(frac_pi_4, pi / 4f64);
        assert_approx_eq!(frac_pi_6, pi / 6f64);
        assert_approx_eq!(frac_pi_8, pi / 8f64);
        assert_approx_eq!(frac_1_pi, 1f64 / pi);
        assert_approx_eq!(frac_2_pi, 2f64 / pi);
        assert_approx_eq!(frac_2_sqrtpi, 2f64 / pi.sqrt());
        assert_approx_eq!(sqrt2, 2f64.sqrt());
        assert_approx_eq!(frac_1_sqrt2, 1f64 / 2f64.sqrt());
        assert_approx_eq!(log2_e, e.log2());
        assert_approx_eq!(log10_e, e.log10());
        assert_approx_eq!(ln_2, 2f64.ln());
        assert_approx_eq!(ln_10, 10f64.ln());
    }

    #[test]
    pub fn test_abs() {
        assert_eq!(INFINITY.abs(), INFINITY);
        assert_eq!(1f64.abs(), 1f64);
        assert_eq!(0f64.abs(), 0f64);
        assert_eq!((-0f64).abs(), 0f64);
        assert_eq!((-1f64).abs(), 1f64);
        assert_eq!(NEG_INFINITY.abs(), INFINITY);
        assert_eq!((1f64/NEG_INFINITY).abs(), 0f64);
        assert!(NAN.abs().is_nan());
    }

    #[test]
    fn test_abs_sub() {
        assert_eq!((-1f64).abs_sub(&1f64), 0f64);
        assert_eq!(1f64.abs_sub(&1f64), 0f64);
        assert_eq!(1f64.abs_sub(&0f64), 1f64);
        assert_eq!(1f64.abs_sub(&-1f64), 2f64);
        assert_eq!(NEG_INFINITY.abs_sub(&0f64), 0f64);
        assert_eq!(INFINITY.abs_sub(&1f64), INFINITY);
        assert_eq!(0f64.abs_sub(&NEG_INFINITY), INFINITY);
        assert_eq!(0f64.abs_sub(&INFINITY), 0f64);
    }

    #[test]
    fn test_abs_sub_nowin() {
        assert!(NAN.abs_sub(&-1f64).is_nan());
        assert!(1f64.abs_sub(&NAN).is_nan());
    }

    #[test]
    fn test_signum() {
        assert_eq!(INFINITY.signum(), 1f64);
        assert_eq!(1f64.signum(), 1f64);
        assert_eq!(0f64.signum(), 1f64);
        assert_eq!((-0f64).signum(), -1f64);
        assert_eq!((-1f64).signum(), -1f64);
        assert_eq!(NEG_INFINITY.signum(), -1f64);
        assert_eq!((1f64/NEG_INFINITY).signum(), -1f64);
        assert!(NAN.signum().is_nan());
    }

    #[test]
    fn test_is_positive() {
        assert!(INFINITY.is_positive());
        assert!(1f64.is_positive());
        assert!(0f64.is_positive());
        assert!(!(-0f64).is_positive());
        assert!(!(-1f64).is_positive());
        assert!(!NEG_INFINITY.is_positive());
        assert!(!(1f64/NEG_INFINITY).is_positive());
        assert!(!NAN.is_positive());
    }

    #[test]
    fn test_is_negative() {
        assert!(!INFINITY.is_negative());
        assert!(!1f64.is_negative());
        assert!(!0f64.is_negative());
        assert!((-0f64).is_negative());
        assert!((-1f64).is_negative());
        assert!(NEG_INFINITY.is_negative());
        assert!((1f64/NEG_INFINITY).is_negative());
        assert!(!NAN.is_negative());
    }

    #[test]
    fn test_is_normal() {
        let nan: f64 = Float::nan();
        let inf: f64 = Float::infinity();
        let neg_inf: f64 = Float::neg_infinity();
        let zero: f64 = Zero::zero();
        let neg_zero: f64 = Float::neg_zero();
        assert!(!nan.is_normal());
        assert!(!inf.is_normal());
        assert!(!neg_inf.is_normal());
        assert!(!zero.is_normal());
        assert!(!neg_zero.is_normal());
        assert!(1f64.is_normal());
        assert!(1e-307f64.is_normal());
        assert!(!1e-308f64.is_normal());
    }

    #[test]
    fn test_classify() {
        let nan: f64 = Float::nan();
        let inf: f64 = Float::infinity();
        let neg_inf: f64 = Float::neg_infinity();
        let zero: f64 = Zero::zero();
        let neg_zero: f64 = Float::neg_zero();
        assert_eq!(nan.classify(), FPNaN);
        assert_eq!(inf.classify(), FPInfinite);
        assert_eq!(neg_inf.classify(), FPInfinite);
        assert_eq!(zero.classify(), FPZero);
        assert_eq!(neg_zero.classify(), FPZero);
        assert_eq!(1e-307f64.classify(), FPNormal);
        assert_eq!(1e-308f64.classify(), FPSubnormal);
    }

    #[test]
    fn test_ldexp() {
        // We have to use from_str until base-2 exponents
        // are supported in floating-point literals
        let f1: f64 = from_str_hex("1p-123").unwrap();
        let f2: f64 = from_str_hex("1p-111").unwrap();
        assert_eq!(Float::ldexp(1f64, -123), f1);
        assert_eq!(Float::ldexp(1f64, -111), f2);

        assert_eq!(Float::ldexp(0f64, -123), 0f64);
        assert_eq!(Float::ldexp(-0f64, -123), -0f64);

        let inf: f64 = Float::infinity();
        let neg_inf: f64 = Float::neg_infinity();
        let nan: f64 = Float::nan();
        assert_eq!(Float::ldexp(inf, -123), inf);
        assert_eq!(Float::ldexp(neg_inf, -123), neg_inf);
        assert!(Float::ldexp(nan, -123).is_nan());
    }

    #[test]
    fn test_frexp() {
        // We have to use from_str until base-2 exponents
        // are supported in floating-point literals
        let f1: f64 = from_str_hex("1p-123").unwrap();
        let f2: f64 = from_str_hex("1p-111").unwrap();
        let (x1, exp1) = f1.frexp();
        let (x2, exp2) = f2.frexp();
        assert_eq!((x1, exp1), (0.5f64, -122));
        assert_eq!((x2, exp2), (0.5f64, -110));
        assert_eq!(Float::ldexp(x1, exp1), f1);
        assert_eq!(Float::ldexp(x2, exp2), f2);

        assert_eq!(0f64.frexp(), (0f64, 0));
        assert_eq!((-0f64).frexp(), (-0f64, 0));
    }

    #[test] #[ignore(cfg(windows))] // FIXME #8755
    fn test_frexp_nowin() {
        let inf: f64 = Float::infinity();
        let neg_inf: f64 = Float::neg_infinity();
        let nan: f64 = Float::nan();
        assert_eq!(match inf.frexp() { (x, _) => x }, inf)
        assert_eq!(match neg_inf.frexp() { (x, _) => x }, neg_inf)
        assert!(match nan.frexp() { (x, _) => x.is_nan() })
    }

    #[test]
    fn test_integer_decode() {
        assert_eq!(3.14159265359f64.integer_decode(), (7074237752028906u64, -51i16, 1i8));
        assert_eq!((-8573.5918555f64).integer_decode(), (4713381968463931u64, -39i16, -1i8));
        assert_eq!(2f64.powf(100.0).integer_decode(), (4503599627370496u64, 48i16, 1i8));
        assert_eq!(0f64.integer_decode(), (0u64, -1075i16, 1i8));
        assert_eq!((-0f64).integer_decode(), (0u64, -1075i16, -1i8));
        assert_eq!(INFINITY.integer_decode(), (4503599627370496u64, 972i16, 1i8));
        assert_eq!(NEG_INFINITY.integer_decode(), (4503599627370496, 972, -1));
        assert_eq!(NAN.integer_decode(), (6755399441055744u64, 972i16, 1i8));
    }
}
