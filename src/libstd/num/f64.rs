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

#![stable]
#![allow(missing_docs)]
#![doc(primitive = "f64")]

use prelude::v1::*;

use intrinsics;
use libc::c_int;
use num::{Float, FpCategory};
use num::strconv;
use num::strconv::ExponentFormat::{ExpNone, ExpDec};
use num::strconv::SignificantDigits::{DigAll, DigMax, DigExact};
use num::strconv::SignFormat::SignNeg;

use core::num;

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

#[stable]
impl Float for f64 {
    // inlined methods from `num::Float`
    #[inline]
    fn nan() -> f64 { num::Float::nan() }
    #[inline]
    fn infinity() -> f64 { num::Float::infinity() }
    #[inline]
    fn neg_infinity() -> f64 { num::Float::neg_infinity() }
    #[inline]
    fn zero() -> f64 { num::Float::zero() }
    #[inline]
    fn neg_zero() -> f64 { num::Float::neg_zero() }
    #[inline]
    fn one() -> f64 { num::Float::one() }


    #[allow(deprecated)]
    #[inline]
    fn mantissa_digits(unused_self: Option<f64>) -> uint {
        num::Float::mantissa_digits(unused_self)
    }
    #[allow(deprecated)]
    #[inline]
    fn digits(unused_self: Option<f64>) -> uint { num::Float::digits(unused_self) }
    #[allow(deprecated)]
    #[inline]
    fn epsilon() -> f64 { num::Float::epsilon() }
    #[allow(deprecated)]
    #[inline]
    fn min_exp(unused_self: Option<f64>) -> int { num::Float::min_exp(unused_self) }
    #[allow(deprecated)]
    #[inline]
    fn max_exp(unused_self: Option<f64>) -> int { num::Float::max_exp(unused_self) }
    #[allow(deprecated)]
    #[inline]
    fn min_10_exp(unused_self: Option<f64>) -> int { num::Float::min_10_exp(unused_self) }
    #[allow(deprecated)]
    #[inline]
    fn max_10_exp(unused_self: Option<f64>) -> int { num::Float::max_10_exp(unused_self) }
    #[allow(deprecated)]
    #[inline]
    fn min_value() -> f64 { num::Float::min_value() }
    #[allow(deprecated)]
    #[inline]
    fn min_pos_value(unused_self: Option<f64>) -> f64 { num::Float::min_pos_value(unused_self) }
    #[allow(deprecated)]
    #[inline]
    fn max_value() -> f64 { num::Float::max_value() }

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
    fn floor(self) -> f64 { num::Float::floor(self) }
    #[inline]
    fn ceil(self) -> f64 { num::Float::ceil(self) }
    #[inline]
    fn round(self) -> f64 { num::Float::round(self) }
    #[inline]
    fn trunc(self) -> f64 { num::Float::trunc(self) }
    #[inline]
    fn fract(self) -> f64 { num::Float::fract(self) }

    #[inline]
    fn abs(self) -> f64 { num::Float::abs(self) }
    #[inline]
    fn signum(self) -> f64 { num::Float::signum(self) }
    #[inline]
    fn is_positive(self) -> bool { num::Float::is_positive(self) }
    #[inline]
    fn is_negative(self) -> bool { num::Float::is_negative(self) }

    #[inline]
    fn mul_add(self, a: f64, b: f64) -> f64 { num::Float::mul_add(self, a, b) }
    #[inline]
    fn recip(self) -> f64 { num::Float::recip(self) }

    #[inline]
    fn powi(self, n: i32) -> f64 { num::Float::powi(self, n) }
    #[inline]
    fn powf(self, n: f64) -> f64 { num::Float::powf(self, n) }

    #[inline]
    fn sqrt(self) -> f64 { num::Float::sqrt(self) }
    #[inline]
    fn rsqrt(self) -> f64 { num::Float::rsqrt(self) }

    #[inline]
    fn exp(self) -> f64 { num::Float::exp(self) }
    #[inline]
    fn exp2(self) -> f64 { num::Float::exp(self) }
    #[inline]
    fn ln(self) -> f64 { num::Float::ln(self) }
    #[inline]
    fn log(self, base: f64) -> f64 { num::Float::log(self, base) }
    #[inline]
    fn log2(self) -> f64 { num::Float::log2(self) }
    #[inline]
    fn log10(self) -> f64 { num::Float::log10(self) }

    #[inline]
    fn to_degrees(self) -> f64 { num::Float::to_degrees(self) }
    #[inline]
    fn to_radians(self) -> f64 { num::Float::to_radians(self) }

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

    /// Returns the next representable floating-point value in the direction of
    /// `other`.
    #[inline]
    fn next_after(self, other: f64) -> f64 {
        unsafe { cmath::nextafter(self, other) }
    }

    #[inline]
    fn max(self, other: f64) -> f64 {
        unsafe { cmath::fmax(self, other) }
    }

    #[inline]
    fn min(self, other: f64) -> f64 {
        unsafe { cmath::fmin(self, other) }
    }

    #[inline]
    fn abs_sub(self, other: f64) -> f64 {
        unsafe { cmath::fdim(self, other) }
    }

    #[inline]
    fn cbrt(self) -> f64 {
        unsafe { cmath::cbrt(self) }
    }

    #[inline]
    fn hypot(self, other: f64) -> f64 {
        unsafe { cmath::hypot(self, other) }
    }

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

    /// Returns the exponential of the number, minus `1`, in a way that is
    /// accurate even if the number is close to zero
    #[inline]
    fn exp_m1(self) -> f64 {
        unsafe { cmath::expm1(self) }
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
#[unstable = "may be removed or relocated"]
pub fn to_string(num: f64) -> String {
    let (r, _) = strconv::float_to_str_common(
        num, 10u, true, SignNeg, DigAll, ExpNone, false);
    r
}

/// Converts a float to a string in hexadecimal format
///
/// # Arguments
///
/// * num - The float value
#[inline]
#[unstable = "may be removed or relocated"]
pub fn to_str_hex(num: f64) -> String {
    let (r, _) = strconv::float_to_str_common(
        num, 16u, true, SignNeg, DigAll, ExpNone, false);
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
#[unstable = "may be removed or relocated"]
pub fn to_str_radix_special(num: f64, rdx: uint) -> (String, bool) {
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
#[unstable = "may be removed or relocated"]
pub fn to_str_exact(num: f64, dig: uint) -> String {
    let (r, _) = strconv::float_to_str_common(
        num, 10u, true, SignNeg, DigExact(dig), ExpNone, false);
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
#[unstable = "may be removed or relocated"]
pub fn to_str_digits(num: f64, dig: uint) -> String {
    let (r, _) = strconv::float_to_str_common(
        num, 10u, true, SignNeg, DigMax(dig), ExpNone, false);
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
#[unstable = "may be removed or relocated"]
pub fn to_str_exp_exact(num: f64, dig: uint, upper: bool) -> String {
    let (r, _) = strconv::float_to_str_common(
        num, 10u, true, SignNeg, DigExact(dig), ExpDec, upper);
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
#[unstable = "may be removed or relocated"]
pub fn to_str_exp_digits(num: f64, dig: uint, upper: bool) -> String {
    let (r, _) = strconv::float_to_str_common(
        num, 10u, true, SignNeg, DigMax(dig), ExpDec, upper);
    r
}

#[cfg(test)]
mod tests {
    use f64::*;
    use num::*;
    use num::FpCategory as Fp;

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
    fn test_num_f64() {
        test_num(10f64, 2f64);
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
        use super::consts;
        let pi: f64 = consts::PI;
        let two_pi: f64 = consts::PI_2;
        let frac_pi_2: f64 = consts::FRAC_PI_2;
        let frac_pi_3: f64 = consts::FRAC_PI_3;
        let frac_pi_4: f64 = consts::FRAC_PI_4;
        let frac_pi_6: f64 = consts::FRAC_PI_6;
        let frac_pi_8: f64 = consts::FRAC_PI_8;
        let frac_1_pi: f64 = consts::FRAC_1_PI;
        let frac_2_pi: f64 = consts::FRAC_2_PI;
        let frac_2_sqrtpi: f64 = consts::FRAC_2_SQRTPI;
        let sqrt2: f64 = consts::SQRT2;
        let frac_1_sqrt2: f64 = consts::FRAC_1_SQRT2;
        let e: f64 = consts::E;
        let log2_e: f64 = consts::LOG2_E;
        let log10_e: f64 = consts::LOG10_E;
        let ln_2: f64 = consts::LN_2;
        let ln_10: f64 = consts::LN_10;

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
        assert_eq!((-1f64).abs_sub(1f64), 0f64);
        assert_eq!(1f64.abs_sub(1f64), 0f64);
        assert_eq!(1f64.abs_sub(0f64), 1f64);
        assert_eq!(1f64.abs_sub(-1f64), 2f64);
        assert_eq!(NEG_INFINITY.abs_sub(0f64), 0f64);
        assert_eq!(INFINITY.abs_sub(1f64), INFINITY);
        assert_eq!(0f64.abs_sub(NEG_INFINITY), INFINITY);
        assert_eq!(0f64.abs_sub(INFINITY), 0f64);
    }

    #[test]
    fn test_abs_sub_nowin() {
        assert!(NAN.abs_sub(-1f64).is_nan());
        assert!(1f64.abs_sub(NAN).is_nan());
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
        let zero: f64 = Float::zero();
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
        let zero: f64 = Float::zero();
        let neg_zero: f64 = Float::neg_zero();
        assert_eq!(nan.classify(), Fp::Nan);
        assert_eq!(inf.classify(), Fp::Infinite);
        assert_eq!(neg_inf.classify(), Fp::Infinite);
        assert_eq!(zero.classify(), Fp::Zero);
        assert_eq!(neg_zero.classify(), Fp::Zero);
        assert_eq!(1e-307f64.classify(), Fp::Normal);
        assert_eq!(1e-308f64.classify(), Fp::Subnormal);
    }

    #[test]
    fn test_ldexp() {
        // We have to use from_str until base-2 exponents
        // are supported in floating-point literals
        let f1: f64 = FromStrRadix::from_str_radix("1p-123", 16).unwrap();
        let f2: f64 = FromStrRadix::from_str_radix("1p-111", 16).unwrap();
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
        let f1: f64 = FromStrRadix::from_str_radix("1p-123", 16).unwrap();
        let f2: f64 = FromStrRadix::from_str_radix("1p-111", 16).unwrap();
        let (x1, exp1) = f1.frexp();
        let (x2, exp2) = f2.frexp();
        assert_eq!((x1, exp1), (0.5f64, -122));
        assert_eq!((x2, exp2), (0.5f64, -110));
        assert_eq!(Float::ldexp(x1, exp1), f1);
        assert_eq!(Float::ldexp(x2, exp2), f2);

        assert_eq!(0f64.frexp(), (0f64, 0));
        assert_eq!((-0f64).frexp(), (-0f64, 0));
    }

    #[test] #[cfg_attr(windows, ignore)] // FIXME #8755
    fn test_frexp_nowin() {
        let inf: f64 = Float::infinity();
        let neg_inf: f64 = Float::neg_infinity();
        let nan: f64 = Float::nan();
        assert_eq!(match inf.frexp() { (x, _) => x }, inf);
        assert_eq!(match neg_inf.frexp() { (x, _) => x }, neg_inf);
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

    #[test]
    fn test_sqrt_domain() {
        assert!(NAN.sqrt().is_nan());
        assert!(NEG_INFINITY.sqrt().is_nan());
        assert!((-1.0f64).sqrt().is_nan());
        assert_eq!((-0.0f64).sqrt(), -0.0);
        assert_eq!(0.0f64.sqrt(), 0.0);
        assert_eq!(1.0f64.sqrt(), 1.0);
        assert_eq!(INFINITY.sqrt(), INFINITY);
    }
}
