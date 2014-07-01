// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for 32-bits floats (`f32` type)

#![experimental]
#![allow(missing_doc)]
#![allow(unsigned_negate)]
#![doc(primitive = "f32")]

use prelude::*;

use from_str::FromStr;
use intrinsics;
use libc::c_int;
use num::strconv;
use num;
use string::String;

pub use core::f32::{RADIX, MANTISSA_DIGITS, DIGITS, EPSILON, MIN_VALUE};
pub use core::f32::{MIN_POS_VALUE, MAX_VALUE, MIN_EXP, MAX_EXP, MIN_10_EXP};
pub use core::f32::{MAX_10_EXP, NAN, INFINITY, NEG_INFINITY};
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

impl FloatMath for f32 {
    /// Constructs a floating point number by multiplying `x` by 2 raised to the
    /// power of `exp`
    #[inline]
    fn ldexp(x: f32, exp: int) -> f32 {
        unsafe { cmath::ldexpf(x, exp as c_int) }
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

//
// Section: String Conversions
//

/// Converts a float to a string
///
/// # Arguments
///
/// * num - The float value
#[inline]
pub fn to_str(num: f32) -> String {
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
pub fn to_str_hex(num: f32) -> String {
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
pub fn to_str_radix_special(num: f32, rdx: uint) -> (String, bool) {
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
pub fn to_str_exact(num: f32, dig: uint) -> String {
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
pub fn to_str_digits(num: f32, dig: uint) -> String {
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
pub fn to_str_exp_exact(num: f32, dig: uint, upper: bool) -> String {
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
pub fn to_str_exp_digits(num: f32, dig: uint, upper: bool) -> String {
    let (r, _) = strconv::float_to_str_common(
        num, 10u, true, strconv::SignNeg, strconv::DigMax(dig), strconv::ExpDec, upper);
    r
}

impl num::ToStrRadix for f32 {
    /// Converts a float to a string in a given radix
    ///
    /// # Arguments
    ///
    /// * num - The float value
    /// * radix - The base to use
    ///
    /// # Failure
    ///
    /// Fails if called on a special value like `inf`, `-inf` or `NaN` due to
    /// possible misinterpretation of the result at higher bases. If those values
    /// are expected, use `to_str_radix_special()` instead.
    #[inline]
    fn to_str_radix(&self, rdx: uint) -> String {
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
pub fn from_str_hex(num: &str) -> Option<f32> {
    strconv::from_str_common(num, 16u, true, true, true,
                             strconv::ExpBin, false, false)
}

impl FromStr for f32 {
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
    /// `None` if the string did not represent a valid number.  Otherwise,
    /// `Some(n)` where `n` is the floating-point number represented by `num`.
    #[inline]
    fn from_str(val: &str) -> Option<f32> {
        strconv::from_str_common(val, 10u, true, true, true,
                                 strconv::ExpDec, false, false)
    }
}

impl num::FromStrRadix for f32 {
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
    fn from_str_radix(val: &str, rdx: uint) -> Option<f32> {
        strconv::from_str_common(val, rdx, true, true, false,
                                 strconv::ExpNone, false, false)
    }
}

#[cfg(test)]
mod tests {
    use f32::*;
    use num::*;
    use num;

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
    fn test_num() {
        num::test_num(10f32, 2f32);
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
        let pi: f32 = Float::pi();
        let two_pi: f32 = Float::two_pi();
        let frac_pi_2: f32 = Float::frac_pi_2();
        let frac_pi_3: f32 = Float::frac_pi_3();
        let frac_pi_4: f32 = Float::frac_pi_4();
        let frac_pi_6: f32 = Float::frac_pi_6();
        let frac_pi_8: f32 = Float::frac_pi_8();
        let frac_1_pi: f32 = Float::frac_1_pi();
        let frac_2_pi: f32 = Float::frac_2_pi();
        let frac_2_sqrtpi: f32 = Float::frac_2_sqrtpi();
        let sqrt2: f32 = Float::sqrt2();
        let frac_1_sqrt2: f32 = Float::frac_1_sqrt2();
        let e: f32 = Float::e();
        let log2_e: f32 = Float::log2_e();
        let log10_e: f32 = Float::log10_e();
        let ln_2: f32 = Float::ln_2();
        let ln_10: f32 = Float::ln_10();

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

    #[test]
    pub fn test_abs() {
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
    fn test_abs_sub() {
        assert_eq!((-1f32).abs_sub(&1f32), 0f32);
        assert_eq!(1f32.abs_sub(&1f32), 0f32);
        assert_eq!(1f32.abs_sub(&0f32), 1f32);
        assert_eq!(1f32.abs_sub(&-1f32), 2f32);
        assert_eq!(NEG_INFINITY.abs_sub(&0f32), 0f32);
        assert_eq!(INFINITY.abs_sub(&1f32), INFINITY);
        assert_eq!(0f32.abs_sub(&NEG_INFINITY), INFINITY);
        assert_eq!(0f32.abs_sub(&INFINITY), 0f32);
    }

    #[test]
    fn test_abs_sub_nowin() {
        assert!(NAN.abs_sub(&-1f32).is_nan());
        assert!(1f32.abs_sub(&NAN).is_nan());
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
    fn test_is_normal() {
        let nan: f32 = Float::nan();
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        let zero: f32 = Zero::zero();
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
        let zero: f32 = Zero::zero();
        let neg_zero: f32 = Float::neg_zero();
        assert_eq!(nan.classify(), FPNaN);
        assert_eq!(inf.classify(), FPInfinite);
        assert_eq!(neg_inf.classify(), FPInfinite);
        assert_eq!(zero.classify(), FPZero);
        assert_eq!(neg_zero.classify(), FPZero);
        assert_eq!(1f32.classify(), FPNormal);
        assert_eq!(1e-37f32.classify(), FPNormal);
        assert_eq!(1e-38f32.classify(), FPSubnormal);
    }

    #[test]
    fn test_ldexp() {
        // We have to use from_str until base-2 exponents
        // are supported in floating-point literals
        let f1: f32 = from_str_hex("1p-123").unwrap();
        let f2: f32 = from_str_hex("1p-111").unwrap();
        assert_eq!(FloatMath::ldexp(1f32, -123), f1);
        assert_eq!(FloatMath::ldexp(1f32, -111), f2);

        assert_eq!(FloatMath::ldexp(0f32, -123), 0f32);
        assert_eq!(FloatMath::ldexp(-0f32, -123), -0f32);

        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        let nan: f32 = Float::nan();
        assert_eq!(FloatMath::ldexp(inf, -123), inf);
        assert_eq!(FloatMath::ldexp(neg_inf, -123), neg_inf);
        assert!(FloatMath::ldexp(nan, -123).is_nan());
    }

    #[test]
    fn test_frexp() {
        // We have to use from_str until base-2 exponents
        // are supported in floating-point literals
        let f1: f32 = from_str_hex("1p-123").unwrap();
        let f2: f32 = from_str_hex("1p-111").unwrap();
        let (x1, exp1) = f1.frexp();
        let (x2, exp2) = f2.frexp();
        assert_eq!((x1, exp1), (0.5f32, -122));
        assert_eq!((x2, exp2), (0.5f32, -110));
        assert_eq!(FloatMath::ldexp(x1, exp1), f1);
        assert_eq!(FloatMath::ldexp(x2, exp2), f2);

        assert_eq!(0f32.frexp(), (0f32, 0));
        assert_eq!((-0f32).frexp(), (-0f32, 0));
    }

    #[test] #[ignore(cfg(windows))] // FIXME #8755
    fn test_frexp_nowin() {
        let inf: f32 = Float::infinity();
        let neg_inf: f32 = Float::neg_infinity();
        let nan: f32 = Float::nan();
        assert_eq!(match inf.frexp() { (x, _) => x }, inf)
        assert_eq!(match neg_inf.frexp() { (x, _) => x }, neg_inf)
        assert!(match nan.frexp() { (x, _) => x.is_nan() })
    }

    #[test]
    fn test_integer_decode() {
        assert_eq!(3.14159265359f32.integer_decode(), (13176795u64, -22i16, 1i8));
        assert_eq!((-8573.5918555f32).integer_decode(), (8779358u64, -10i16, -1i8));
        assert_eq!(2f32.powf(100.0).integer_decode(), (8388608u64, 77i16, 1i8));
        assert_eq!(0f32.integer_decode(), (0u64, -150i16, 1i8));
        assert_eq!((-0f32).integer_decode(), (0u64, -150i16, -1i8));
        assert_eq!(INFINITY.integer_decode(), (8388608u64, 105i16, 1i8));
        assert_eq!(NEG_INFINITY.integer_decode(), (8388608u64, 105i16, -1i8));
        assert_eq!(NAN.integer_decode(), (12582912u64, 105i16, 1i8));
    }
}
