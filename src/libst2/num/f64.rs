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

use prelude::*;

use intrinsics;
use libc::c_int;
use num::{Float, FloatMath};
use num::strconv;

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

#[unstable = "trait is unstable"]
impl FloatMath for f64 {
    /// Constructs a floating point number by multiplying `x` by 2 raised to the
    /// power of `exp`
    #[inline]
    fn ldexp(x: f64, exp: int) -> f64 { unimplemented!() }

    /// Breaks the number into a normalized fraction and a base-2 exponent,
    /// satisfying:
    ///
    /// - `self = x * pow(2, exp)`
    /// - `0.5 <= abs(x) < 1.0`
    #[inline]
    fn frexp(self) -> (f64, int) { unimplemented!() }

    /// Returns the next representable floating-point value in the direction of
    /// `other`.
    #[inline]
    fn next_after(self, other: f64) -> f64 { unimplemented!() }

    #[inline]
    fn max(self, other: f64) -> f64 { unimplemented!() }

    #[inline]
    fn min(self, other: f64) -> f64 { unimplemented!() }

    #[inline]
    fn abs_sub(self, other: f64) -> f64 { unimplemented!() }

    #[inline]
    fn cbrt(self) -> f64 { unimplemented!() }

    #[inline]
    fn hypot(self, other: f64) -> f64 { unimplemented!() }

    #[inline]
    fn sin(self) -> f64 { unimplemented!() }

    #[inline]
    fn cos(self) -> f64 { unimplemented!() }

    #[inline]
    fn tan(self) -> f64 { unimplemented!() }

    #[inline]
    fn asin(self) -> f64 { unimplemented!() }

    #[inline]
    fn acos(self) -> f64 { unimplemented!() }

    #[inline]
    fn atan(self) -> f64 { unimplemented!() }

    #[inline]
    fn atan2(self, other: f64) -> f64 { unimplemented!() }

    /// Simultaneously computes the sine and cosine of the number
    #[inline]
    fn sin_cos(self) -> (f64, f64) { unimplemented!() }

    /// Returns the exponential of the number, minus `1`, in a way that is
    /// accurate even if the number is close to zero
    #[inline]
    fn exp_m1(self) -> f64 { unimplemented!() }

    /// Returns the natural logarithm of the number plus `1` (`ln(1+n)`) more
    /// accurately than if the operations were performed separately
    #[inline]
    fn ln_1p(self) -> f64 { unimplemented!() }

    #[inline]
    fn sinh(self) -> f64 { unimplemented!() }

    #[inline]
    fn cosh(self) -> f64 { unimplemented!() }

    #[inline]
    fn tanh(self) -> f64 { unimplemented!() }

    /// Inverse hyperbolic sine
    ///
    /// # Returns
    ///
    /// - on success, the inverse hyperbolic sine of `self` will be returned
    /// - `self` if `self` is `0.0`, `-0.0`, `INFINITY`, or `NEG_INFINITY`
    /// - `NAN` if `self` is `NAN`
    #[inline]
    fn asinh(self) -> f64 { unimplemented!() }

    /// Inverse hyperbolic cosine
    ///
    /// # Returns
    ///
    /// - on success, the inverse hyperbolic cosine of `self` will be returned
    /// - `INFINITY` if `self` is `INFINITY`
    /// - `NAN` if `self` is `NAN` or `self < 1.0` (including `NEG_INFINITY`)
    #[inline]
    fn acosh(self) -> f64 { unimplemented!() }

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
    fn atanh(self) -> f64 { unimplemented!() }
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
#[experimental = "may be removed or relocated"]
pub fn to_string(num: f64) -> String { unimplemented!() }

/// Converts a float to a string in hexadecimal format
///
/// # Arguments
///
/// * num - The float value
#[inline]
#[experimental = "may be removed or relocated"]
pub fn to_str_hex(num: f64) -> String { unimplemented!() }

/// Converts a float to a string in a given radix, and a flag indicating
/// whether it's a special value
///
/// # Arguments
///
/// * num - The float value
/// * radix - The base to use
#[inline]
#[experimental = "may be removed or relocated"]
pub fn to_str_radix_special(num: f64, rdx: uint) -> (String, bool) { unimplemented!() }

/// Converts a float to a string with exactly the number of
/// provided significant digits
///
/// # Arguments
///
/// * num - The float value
/// * digits - The number of significant digits
#[inline]
#[experimental = "may be removed or relocated"]
pub fn to_str_exact(num: f64, dig: uint) -> String { unimplemented!() }

/// Converts a float to a string with a maximum number of
/// significant digits
///
/// # Arguments
///
/// * num - The float value
/// * digits - The number of significant digits
#[inline]
#[experimental = "may be removed or relocated"]
pub fn to_str_digits(num: f64, dig: uint) -> String { unimplemented!() }

/// Converts a float to a string using the exponential notation with exactly the number of
/// provided digits after the decimal point in the significand
///
/// # Arguments
///
/// * num - The float value
/// * digits - The number of digits after the decimal point
/// * upper - Use `E` instead of `e` for the exponent sign
#[inline]
#[experimental = "may be removed or relocated"]
pub fn to_str_exp_exact(num: f64, dig: uint, upper: bool) -> String { unimplemented!() }

/// Converts a float to a string using the exponential notation with the maximum number of
/// digits after the decimal point in the significand
///
/// # Arguments
///
/// * num - The float value
/// * digits - The number of digits after the decimal point
/// * upper - Use `E` instead of `e` for the exponent sign
#[inline]
#[experimental = "may be removed or relocated"]
pub fn to_str_exp_digits(num: f64, dig: uint, upper: bool) -> String { unimplemented!() }
