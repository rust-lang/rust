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

#![stable]
#![allow(missing_docs)]
#![allow(unsigned_negation)]
#![doc(primitive = "f32")]

use prelude::*;

use intrinsics;
use libc::c_int;
use num::{Float, FloatMath};
use num::strconv;

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

#[unstable = "trait is unstable"]
impl FloatMath for f32 {
    /// Constructs a floating point number by multiplying `x` by 2 raised to the
    /// power of `exp`
    #[inline]
    fn ldexp(x: f32, exp: int) -> f32 { unimplemented!() }

    /// Breaks the number into a normalized fraction and a base-2 exponent,
    /// satisfying:
    ///
    /// - `self = x * pow(2, exp)`
    /// - `0.5 <= abs(x) < 1.0`
    #[inline]
    fn frexp(self) -> (f32, int) { unimplemented!() }

    /// Returns the next representable floating-point value in the direction of
    /// `other`.
    #[inline]
    fn next_after(self, other: f32) -> f32 { unimplemented!() }

    #[inline]
    fn max(self, other: f32) -> f32 { unimplemented!() }

    #[inline]
    fn min(self, other: f32) -> f32 { unimplemented!() }

    #[inline]
    fn abs_sub(self, other: f32) -> f32 { unimplemented!() }

    #[inline]
    fn cbrt(self) -> f32 { unimplemented!() }

    #[inline]
    fn hypot(self, other: f32) -> f32 { unimplemented!() }

    #[inline]
    fn sin(self) -> f32 { unimplemented!() }

    #[inline]
    fn cos(self) -> f32 { unimplemented!() }

    #[inline]
    fn tan(self) -> f32 { unimplemented!() }

    #[inline]
    fn asin(self) -> f32 { unimplemented!() }

    #[inline]
    fn acos(self) -> f32 { unimplemented!() }

    #[inline]
    fn atan(self) -> f32 { unimplemented!() }

    #[inline]
    fn atan2(self, other: f32) -> f32 { unimplemented!() }

    /// Simultaneously computes the sine and cosine of the number
    #[inline]
    fn sin_cos(self) -> (f32, f32) { unimplemented!() }

    /// Returns the exponential of the number, minus `1`, in a way that is
    /// accurate even if the number is close to zero
    #[inline]
    fn exp_m1(self) -> f32 { unimplemented!() }

    /// Returns the natural logarithm of the number plus `1` (`ln(1+n)`) more
    /// accurately than if the operations were performed separately
    #[inline]
    fn ln_1p(self) -> f32 { unimplemented!() }

    #[inline]
    fn sinh(self) -> f32 { unimplemented!() }

    #[inline]
    fn cosh(self) -> f32 { unimplemented!() }

    #[inline]
    fn tanh(self) -> f32 { unimplemented!() }

    /// Inverse hyperbolic sine
    ///
    /// # Returns
    ///
    /// - on success, the inverse hyperbolic sine of `self` will be returned
    /// - `self` if `self` is `0.0`, `-0.0`, `INFINITY`, or `NEG_INFINITY`
    /// - `NAN` if `self` is `NAN`
    #[inline]
    fn asinh(self) -> f32 { unimplemented!() }

    /// Inverse hyperbolic cosine
    ///
    /// # Returns
    ///
    /// - on success, the inverse hyperbolic cosine of `self` will be returned
    /// - `INFINITY` if `self` is `INFINITY`
    /// - `NAN` if `self` is `NAN` or `self < 1.0` (including `NEG_INFINITY`)
    #[inline]
    fn acosh(self) -> f32 { unimplemented!() }

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
    fn atanh(self) -> f32 { unimplemented!() }
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
pub fn to_string(num: f32) -> String { unimplemented!() }

/// Converts a float to a string in hexadecimal format
///
/// # Arguments
///
/// * num - The float value
#[inline]
#[experimental = "may be removed or relocated"]
pub fn to_str_hex(num: f32) -> String { unimplemented!() }

/// Converts a float to a string in a given radix, and a flag indicating
/// whether it's a special value
///
/// # Arguments
///
/// * num - The float value
/// * radix - The base to use
#[inline]
#[experimental = "may be removed or relocated"]
pub fn to_str_radix_special(num: f32, rdx: uint) -> (String, bool) { unimplemented!() }

/// Converts a float to a string with exactly the number of
/// provided significant digits
///
/// # Arguments
///
/// * num - The float value
/// * digits - The number of significant digits
#[inline]
#[experimental = "may be removed or relocated"]
pub fn to_str_exact(num: f32, dig: uint) -> String { unimplemented!() }

/// Converts a float to a string with a maximum number of
/// significant digits
///
/// # Arguments
///
/// * num - The float value
/// * digits - The number of significant digits
#[inline]
#[experimental = "may be removed or relocated"]
pub fn to_str_digits(num: f32, dig: uint) -> String { unimplemented!() }

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
pub fn to_str_exp_exact(num: f32, dig: uint, upper: bool) -> String { unimplemented!() }

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
pub fn to_str_exp_digits(num: f32, dig: uint, upper: bool) -> String { unimplemented!() }
