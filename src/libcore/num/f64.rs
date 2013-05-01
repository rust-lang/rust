// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for `f64`

use from_str;
use libc::c_int;
use num::{Zero, One, strconv};
use prelude::*;

pub use cmath::c_double_targ_consts::*;
pub use cmp::{min, max};

// An inner module is required to get the #[inline(always)] attribute on the
// functions.
pub use self::delegated::*;

macro_rules! delegate(
    (
        $(
            fn $name:ident(
                $(
                    $arg:ident : $arg_ty:ty
                ),*
            ) -> $rv:ty = $bound_name:path
        ),*
    ) => (
        mod delegated {
            use cmath::c_double_utils;
            use libc::{c_double, c_int};
            use unstable::intrinsics;

            $(
                #[inline(always)]
                pub fn $name($( $arg : $arg_ty ),*) -> $rv {
                    unsafe {
                        $bound_name($( $arg ),*)
                    }
                }
            )*
        }
    )
)

delegate!(
    // intrinsics
    fn abs(n: f64) -> f64 = intrinsics::fabsf64,
    fn cos(n: f64) -> f64 = intrinsics::cosf64,
    fn exp(n: f64) -> f64 = intrinsics::expf64,
    fn exp2(n: f64) -> f64 = intrinsics::exp2f64,
    fn floor(x: f64) -> f64 = intrinsics::floorf64,
    fn ln(n: f64) -> f64 = intrinsics::logf64,
    fn log10(n: f64) -> f64 = intrinsics::log10f64,
    fn log2(n: f64) -> f64 = intrinsics::log2f64,
    fn mul_add(a: f64, b: f64, c: f64) -> f64 = intrinsics::fmaf64,
    fn pow(n: f64, e: f64) -> f64 = intrinsics::powf64,
    fn powi(n: f64, e: c_int) -> f64 = intrinsics::powif64,
    fn sin(n: f64) -> f64 = intrinsics::sinf64,
    fn sqrt(n: f64) -> f64 = intrinsics::sqrtf64,

    // LLVM 3.3 required to use intrinsics for these four
    fn ceil(n: c_double) -> c_double = c_double_utils::ceil,
    fn trunc(n: c_double) -> c_double = c_double_utils::trunc,
    /*
    fn ceil(n: f64) -> f64 = intrinsics::ceilf64,
    fn trunc(n: f64) -> f64 = intrinsics::truncf64,
    fn rint(n: c_double) -> c_double = intrinsics::rintf64,
    fn nearbyint(n: c_double) -> c_double = intrinsics::nearbyintf64,
    */

    // cmath
    fn acos(n: c_double) -> c_double = c_double_utils::acos,
    fn asin(n: c_double) -> c_double = c_double_utils::asin,
    fn atan(n: c_double) -> c_double = c_double_utils::atan,
    fn atan2(a: c_double, b: c_double) -> c_double = c_double_utils::atan2,
    fn cbrt(n: c_double) -> c_double = c_double_utils::cbrt,
    fn copysign(x: c_double, y: c_double) -> c_double = c_double_utils::copysign,
    fn cosh(n: c_double) -> c_double = c_double_utils::cosh,
    fn erf(n: c_double) -> c_double = c_double_utils::erf,
    fn erfc(n: c_double) -> c_double = c_double_utils::erfc,
    fn expm1(n: c_double) -> c_double = c_double_utils::expm1,
    fn abs_sub(a: c_double, b: c_double) -> c_double = c_double_utils::abs_sub,
    fn fmax(a: c_double, b: c_double) -> c_double = c_double_utils::fmax,
    fn fmin(a: c_double, b: c_double) -> c_double = c_double_utils::fmin,
    fn next_after(x: c_double, y: c_double) -> c_double = c_double_utils::next_after,
    fn frexp(n: c_double, value: &mut c_int) -> c_double = c_double_utils::frexp,
    fn hypot(x: c_double, y: c_double) -> c_double = c_double_utils::hypot,
    fn ldexp(x: c_double, n: c_int) -> c_double = c_double_utils::ldexp,
    fn lgamma(n: c_double, sign: &mut c_int) -> c_double = c_double_utils::lgamma,
    fn log_radix(n: c_double) -> c_double = c_double_utils::log_radix,
    fn ln1p(n: c_double) -> c_double = c_double_utils::ln1p,
    fn ilog_radix(n: c_double) -> c_int = c_double_utils::ilog_radix,
    fn modf(n: c_double, iptr: &mut c_double) -> c_double = c_double_utils::modf,
    fn round(n: c_double) -> c_double = c_double_utils::round,
    fn ldexp_radix(n: c_double, i: c_int) -> c_double = c_double_utils::ldexp_radix,
    fn sinh(n: c_double) -> c_double = c_double_utils::sinh,
    fn tan(n: c_double) -> c_double = c_double_utils::tan,
    fn tanh(n: c_double) -> c_double = c_double_utils::tanh,
    fn tgamma(n: c_double) -> c_double = c_double_utils::tgamma,
    fn j0(n: c_double) -> c_double = c_double_utils::j0,
    fn j1(n: c_double) -> c_double = c_double_utils::j1,
    fn jn(i: c_int, n: c_double) -> c_double = c_double_utils::jn,
    fn y0(n: c_double) -> c_double = c_double_utils::y0,
    fn y1(n: c_double) -> c_double = c_double_utils::y1,
    fn yn(i: c_int, n: c_double) -> c_double = c_double_utils::yn
)

// FIXME (#1433): obtain these in a different way

// These are not defined inside consts:: for consistency with
// the integer types

pub static radix: uint = 2u;

pub static mantissa_digits: uint = 53u;
pub static digits: uint = 15u;

pub static epsilon: f64 = 2.2204460492503131e-16_f64;

pub static min_value: f64 = 2.2250738585072014e-308_f64;
pub static max_value: f64 = 1.7976931348623157e+308_f64;

pub static min_exp: int = -1021;
pub static max_exp: int = 1024;

pub static min_10_exp: int = -307;
pub static max_10_exp: int = 308;

pub static NaN: f64 = 0.0_f64/0.0_f64;

pub static infinity: f64 = 1.0_f64/0.0_f64;

pub static neg_infinity: f64 = -1.0_f64/0.0_f64;

#[inline(always)]
pub fn add(x: f64, y: f64) -> f64 { return x + y; }

#[inline(always)]
pub fn sub(x: f64, y: f64) -> f64 { return x - y; }

#[inline(always)]
pub fn mul(x: f64, y: f64) -> f64 { return x * y; }

#[inline(always)]
pub fn div(x: f64, y: f64) -> f64 { return x / y; }

#[inline(always)]
pub fn rem(x: f64, y: f64) -> f64 { return x % y; }

#[inline(always)]
pub fn lt(x: f64, y: f64) -> bool { return x < y; }

#[inline(always)]
pub fn le(x: f64, y: f64) -> bool { return x <= y; }

#[inline(always)]
pub fn eq(x: f64, y: f64) -> bool { return x == y; }

#[inline(always)]
pub fn ne(x: f64, y: f64) -> bool { return x != y; }

#[inline(always)]
pub fn ge(x: f64, y: f64) -> bool { return x >= y; }

#[inline(always)]
pub fn gt(x: f64, y: f64) -> bool { return x > y; }


// FIXME (#1999): add is_normal, is_subnormal, and fpclassify

/* Module: consts */
pub mod consts {
    // FIXME (requires Issue #1433 to fix): replace with mathematical
    // constants from cmath.
    /// Archimedes' constant
    pub static pi: f64 = 3.14159265358979323846264338327950288_f64;

    /// pi/2.0
    pub static frac_pi_2: f64 = 1.57079632679489661923132169163975144_f64;

    /// pi/4.0
    pub static frac_pi_4: f64 = 0.785398163397448309615660845819875721_f64;

    /// 1.0/pi
    pub static frac_1_pi: f64 = 0.318309886183790671537767526745028724_f64;

    /// 2.0/pi
    pub static frac_2_pi: f64 = 0.636619772367581343075535053490057448_f64;

    /// 2.0/sqrt(pi)
    pub static frac_2_sqrtpi: f64 = 1.12837916709551257389615890312154517_f64;

    /// sqrt(2.0)
    pub static sqrt2: f64 = 1.41421356237309504880168872420969808_f64;

    /// 1.0/sqrt(2.0)
    pub static frac_1_sqrt2: f64 = 0.707106781186547524400844362104849039_f64;

    /// Euler's number
    pub static e: f64 = 2.71828182845904523536028747135266250_f64;

    /// log2(e)
    pub static log2_e: f64 = 1.44269504088896340735992468100189214_f64;

    /// log10(e)
    pub static log10_e: f64 = 0.434294481903251827651128918916605082_f64;

    /// ln(2.0)
    pub static ln_2: f64 = 0.693147180559945309417232121458176568_f64;

    /// ln(10.0)
    pub static ln_10: f64 = 2.30258509299404568401799145468436421_f64;
}

#[inline(always)]
pub fn logarithm(n: f64, b: f64) -> f64 {
    return log2(n) / log2(b);
}

impl Num for f64 {}

#[cfg(notest)]
impl Eq for f64 {
    #[inline(always)]
    fn eq(&self, other: &f64) -> bool { (*self) == (*other) }
    #[inline(always)]
    fn ne(&self, other: &f64) -> bool { (*self) != (*other) }
}

#[cfg(notest)]
impl Ord for f64 {
    #[inline(always)]
    fn lt(&self, other: &f64) -> bool { (*self) < (*other) }
    #[inline(always)]
    fn le(&self, other: &f64) -> bool { (*self) <= (*other) }
    #[inline(always)]
    fn ge(&self, other: &f64) -> bool { (*self) >= (*other) }
    #[inline(always)]
    fn gt(&self, other: &f64) -> bool { (*self) > (*other) }
}

impl Orderable for f64 {
    /// Returns `NaN` if either of the numbers are `NaN`.
    #[inline(always)]
    fn min(&self, other: &f64) -> f64 {
        if self.is_NaN() || other.is_NaN() { Float::NaN() } else { fmin(*self, *other) }
    }

    /// Returns `NaN` if either of the numbers are `NaN`.
    #[inline(always)]
    fn max(&self, other: &f64) -> f64 {
        if self.is_NaN() || other.is_NaN() { Float::NaN() } else { fmax(*self, *other) }
    }

    /// Returns the number constrained within the range `mn <= self <= mx`.
    /// If any of the numbers are `NaN` then `NaN` is returned.
    #[inline(always)]
    fn clamp(&self, mn: &f64, mx: &f64) -> f64 {
        if self.is_NaN() { *self }
        else if !(*self <= *mx) { *mx }
        else if !(*self >= *mn) { *mn }
        else { *self }
    }
}

impl Zero for f64 {
    #[inline(always)]
    fn zero() -> f64 { 0.0 }

    /// Returns true if the number is equal to either `0.0` or `-0.0`
    #[inline(always)]
    fn is_zero(&self) -> bool { *self == 0.0 || *self == -0.0 }
}

impl One for f64 {
    #[inline(always)]
    fn one() -> f64 { 1.0 }
}

#[cfg(notest)]
impl Add<f64,f64> for f64 {
    fn add(&self, other: &f64) -> f64 { *self + *other }
}
#[cfg(notest)]
impl Sub<f64,f64> for f64 {
    fn sub(&self, other: &f64) -> f64 { *self - *other }
}
#[cfg(notest)]
impl Mul<f64,f64> for f64 {
    fn mul(&self, other: &f64) -> f64 { *self * *other }
}
#[cfg(notest)]
impl Div<f64,f64> for f64 {
    fn div(&self, other: &f64) -> f64 { *self / *other }
}
#[cfg(stage0,notest)]
impl Modulo<f64,f64> for f64 {
    fn modulo(&self, other: &f64) -> f64 { *self % *other }
}
#[cfg(not(stage0),notest)]
impl Rem<f64,f64> for f64 {
    #[inline(always)]
    fn rem(&self, other: &f64) -> f64 { *self % *other }
}
#[cfg(notest)]
impl Neg<f64> for f64 {
    fn neg(&self) -> f64 { -*self }
}

impl Signed for f64 {
    /// Computes the absolute value. Returns `NaN` if the number is `NaN`.
    #[inline(always)]
    fn abs(&self) -> f64 { abs(*self) }

    ///
    /// # Returns
    ///
    /// - `1.0` if the number is positive, `+0.0` or `infinity`
    /// - `-1.0` if the number is negative, `-0.0` or `neg_infinity`
    /// - `NaN` if the number is NaN
    ///
    #[inline(always)]
    fn signum(&self) -> f64 {
        if self.is_NaN() { NaN } else { copysign(1.0, *self) }
    }

    /// Returns `true` if the number is positive, including `+0.0` and `infinity`
    #[inline(always)]
    fn is_positive(&self) -> bool { *self > 0.0 || (1.0 / *self) == infinity }

    /// Returns `true` if the number is negative, including `-0.0` and `neg_infinity`
    #[inline(always)]
    fn is_negative(&self) -> bool { *self < 0.0 || (1.0 / *self) == neg_infinity }
}

impl Round for f64 {
    /// Round half-way cases toward `neg_infinity`
    #[inline(always)]
    fn floor(&self) -> f64 { floor(*self) }

    /// Round half-way cases toward `infinity`
    #[inline(always)]
    fn ceil(&self) -> f64 { ceil(*self) }

    /// Round half-way cases away from `0.0`
    #[inline(always)]
    fn round(&self) -> f64 { round(*self) }

    /// The integer part of the number (rounds towards `0.0`)
    #[inline(always)]
    fn trunc(&self) -> f64 { trunc(*self) }

    ///
    /// The fractional part of the number, satisfying:
    ///
    /// ~~~
    /// assert!(x == trunc(x) + fract(x))
    /// ~~~
    ///
    #[inline(always)]
    fn fract(&self) -> f64 { *self - self.trunc() }
}

impl Fractional for f64 {
    /// The reciprocal (multiplicative inverse) of the number
    #[inline(always)]
    fn recip(&self) -> f64 { 1.0 / *self }
}

impl Algebraic for f64 {
    #[inline(always)]
    fn pow(&self, n: f64) -> f64 { pow(*self, n) }

    #[inline(always)]
    fn sqrt(&self) -> f64 { sqrt(*self) }

    #[inline(always)]
    fn rsqrt(&self) -> f64 { self.sqrt().recip() }

    #[inline(always)]
    fn cbrt(&self) -> f64 { cbrt(*self) }

    #[inline(always)]
    fn hypot(&self, other: f64) -> f64 { hypot(*self, other) }
}

impl Trigonometric for f64 {
    #[inline(always)]
    fn sin(&self) -> f64 { sin(*self) }

    #[inline(always)]
    fn cos(&self) -> f64 { cos(*self) }

    #[inline(always)]
    fn tan(&self) -> f64 { tan(*self) }

    #[inline(always)]
    fn asin(&self) -> f64 { asin(*self) }

    #[inline(always)]
    fn acos(&self) -> f64 { acos(*self) }

    #[inline(always)]
    fn atan(&self) -> f64 { atan(*self) }

    #[inline(always)]
    fn atan2(&self, other: f64) -> f64 { atan2(*self, other) }
}

impl Exponential for f64 {
    #[inline(always)]
    fn exp(&self) -> f64 { exp(*self) }

    #[inline(always)]
    fn exp2(&self) -> f64 { exp2(*self) }

    #[inline(always)]
    fn expm1(&self) -> f64 { expm1(*self) }

    #[inline(always)]
    fn log(&self) -> f64 { ln(*self) }

    #[inline(always)]
    fn log2(&self) -> f64 { log2(*self) }

    #[inline(always)]
    fn log10(&self) -> f64 { log10(*self) }
}

impl Hyperbolic for f64 {
    #[inline(always)]
    fn sinh(&self) -> f64 { sinh(*self) }

    #[inline(always)]
    fn cosh(&self) -> f64 { cosh(*self) }

    #[inline(always)]
    fn tanh(&self) -> f64 { tanh(*self) }
}

impl Real for f64 {
    /// Archimedes' constant
    #[inline(always)]
    fn pi() -> f64 { 3.14159265358979323846264338327950288 }

    /// 2.0 * pi
    #[inline(always)]
    fn two_pi() -> f64 { 6.28318530717958647692528676655900576 }

    /// pi / 2.0
    #[inline(always)]
    fn frac_pi_2() -> f64 { 1.57079632679489661923132169163975144 }

    /// pi / 3.0
    #[inline(always)]
    fn frac_pi_3() -> f64 { 1.04719755119659774615421446109316763 }

    /// pi / 4.0
    #[inline(always)]
    fn frac_pi_4() -> f64 { 0.785398163397448309615660845819875721 }

    /// pi / 6.0
    #[inline(always)]
    fn frac_pi_6() -> f64 { 0.52359877559829887307710723054658381 }

    /// pi / 8.0
    #[inline(always)]
    fn frac_pi_8() -> f64 { 0.39269908169872415480783042290993786 }

    /// 1.0 / pi
    #[inline(always)]
    fn frac_1_pi() -> f64 { 0.318309886183790671537767526745028724 }

    /// 2.0 / pi
    #[inline(always)]
    fn frac_2_pi() -> f64 { 0.636619772367581343075535053490057448 }

    /// 2.0 / sqrt(pi)
    #[inline(always)]
    fn frac_2_sqrtpi() -> f64 { 1.12837916709551257389615890312154517 }

    /// sqrt(2.0)
    #[inline(always)]
    fn sqrt2() -> f64 { 1.41421356237309504880168872420969808 }

    /// 1.0 / sqrt(2.0)
    #[inline(always)]
    fn frac_1_sqrt2() -> f64 { 0.707106781186547524400844362104849039 }

    /// Euler's number
    #[inline(always)]
    fn e() -> f64 { 2.71828182845904523536028747135266250 }

    /// log2(e)
    #[inline(always)]
    fn log2_e() -> f64 { 1.44269504088896340735992468100189214 }

    /// log10(e)
    #[inline(always)]
    fn log10_e() -> f64 { 0.434294481903251827651128918916605082 }

    /// log(2.0)
    #[inline(always)]
    fn log_2() -> f64 { 0.693147180559945309417232121458176568 }

    /// log(10.0)
    #[inline(always)]
    fn log_10() -> f64 { 2.30258509299404568401799145468436421 }

    /// Converts to degrees, assuming the number is in radians
    #[inline(always)]
    fn to_degrees(&self) -> f64 { *self * (180.0 / Real::pi::<f64>()) }

    /// Converts to radians, assuming the number is in degrees
    #[inline(always)]
    fn to_radians(&self) -> f64 { *self * (Real::pi::<f64>() / 180.0) }
}

impl RealExt for f64 {
    #[inline(always)]
    fn lgamma(&self) -> (int, f64) {
        let mut sign = 0;
        let result = lgamma(*self, &mut sign);
        (sign as int, result)
    }

    #[inline(always)]
    fn tgamma(&self) -> f64 { tgamma(*self) }

    #[inline(always)]
    fn j0(&self) -> f64 { j0(*self) }

    #[inline(always)]
    fn j1(&self) -> f64 { j1(*self) }

    #[inline(always)]
    fn jn(&self, n: int) -> f64 { jn(n as c_int, *self) }

    #[inline(always)]
    fn y0(&self) -> f64 { y0(*self) }

    #[inline(always)]
    fn y1(&self) -> f64 { y1(*self) }

    #[inline(always)]
    fn yn(&self, n: int) -> f64 { yn(n as c_int, *self) }
}

impl Bounded for f64 {
    #[inline(always)]
    fn min_value() -> f64 { 2.2250738585072014e-308 }

    #[inline(always)]
    fn max_value() -> f64 { 1.7976931348623157e+308 }
}

impl Primitive for f64 {
    #[inline(always)]
    fn bits() -> uint { 64 }

    #[inline(always)]
    fn bytes() -> uint { Primitive::bits::<f64>() / 8 }
}

impl Float for f64 {
    #[inline(always)]
    fn NaN() -> f64 { 0.0 / 0.0 }

    #[inline(always)]
    fn infinity() -> f64 { 1.0 / 0.0 }

    #[inline(always)]
    fn neg_infinity() -> f64 { -1.0 / 0.0 }

    #[inline(always)]
    fn neg_zero() -> f64 { -0.0 }

    #[inline(always)]
    fn is_NaN(&self) -> bool { *self != *self }

    /// Returns `true` if the number is infinite
    #[inline(always)]
    fn is_infinite(&self) -> bool {
        *self == Float::infinity() || *self == Float::neg_infinity()
    }

    /// Returns `true` if the number is finite
    #[inline(always)]
    fn is_finite(&self) -> bool {
        !(self.is_NaN() || self.is_infinite())
    }

    #[inline(always)]
    fn mantissa_digits() -> uint { 53 }

    #[inline(always)]
    fn digits() -> uint { 15 }

    #[inline(always)]
    fn epsilon() -> f64 { 2.2204460492503131e-16 }

    #[inline(always)]
    fn min_exp() -> int { -1021 }

    #[inline(always)]
    fn max_exp() -> int { 1024 }

    #[inline(always)]
    fn min_10_exp() -> int { -307 }

    #[inline(always)]
    fn max_10_exp() -> int { 308 }

    ///
    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding error. This
    /// produces a more accurate result with better performance than a separate multiplication
    /// operation followed by an add.
    ///
    #[inline(always)]
    fn mul_add(&self, a: f64, b: f64) -> f64 {
        mul_add(*self, a, b)
    }

    /// Returns the next representable floating-point value in the direction of `other`
    #[inline(always)]
    fn next_after(&self, other: f64) -> f64 {
        next_after(*self, other)
    }
}

//
// Section: String Conversions
//

///
/// Converts a float to a string
///
/// # Arguments
///
/// * num - The float value
///
#[inline(always)]
pub fn to_str(num: f64) -> ~str {
    let (r, _) = strconv::to_str_common(
        &num, 10u, true, strconv::SignNeg, strconv::DigAll);
    r
}

///
/// Converts a float to a string in hexadecimal format
///
/// # Arguments
///
/// * num - The float value
///
#[inline(always)]
pub fn to_str_hex(num: f64) -> ~str {
    let (r, _) = strconv::to_str_common(
        &num, 16u, true, strconv::SignNeg, strconv::DigAll);
    r
}

///
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
///
#[inline(always)]
pub fn to_str_radix(num: f64, rdx: uint) -> ~str {
    let (r, special) = strconv::to_str_common(
        &num, rdx, true, strconv::SignNeg, strconv::DigAll);
    if special { fail!(~"number has a special value, \
                      try to_str_radix_special() if those are expected") }
    r
}

///
/// Converts a float to a string in a given radix, and a flag indicating
/// whether it's a special value
///
/// # Arguments
///
/// * num - The float value
/// * radix - The base to use
///
#[inline(always)]
pub fn to_str_radix_special(num: f64, rdx: uint) -> (~str, bool) {
    strconv::to_str_common(&num, rdx, true,
                           strconv::SignNeg, strconv::DigAll)
}

///
/// Converts a float to a string with exactly the number of
/// provided significant digits
///
/// # Arguments
///
/// * num - The float value
/// * digits - The number of significant digits
///
#[inline(always)]
pub fn to_str_exact(num: f64, dig: uint) -> ~str {
    let (r, _) = strconv::to_str_common(
        &num, 10u, true, strconv::SignNeg, strconv::DigExact(dig));
    r
}

///
/// Converts a float to a string with a maximum number of
/// significant digits
///
/// # Arguments
///
/// * num - The float value
/// * digits - The number of significant digits
///
#[inline(always)]
pub fn to_str_digits(num: f64, dig: uint) -> ~str {
    let (r, _) = strconv::to_str_common(
        &num, 10u, true, strconv::SignNeg, strconv::DigMax(dig));
    r
}

impl to_str::ToStr for f64 {
    #[inline(always)]
    fn to_str(&self) -> ~str { to_str_digits(*self, 8) }
}

impl num::ToStrRadix for f64 {
    #[inline(always)]
    fn to_str_radix(&self, rdx: uint) -> ~str {
        to_str_radix(*self, rdx)
    }
}

///
/// Convert a string in base 10 to a float.
/// Accepts a optional decimal exponent.
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
///
#[inline(always)]
pub fn from_str(num: &str) -> Option<f64> {
    strconv::from_str_common(num, 10u, true, true, true,
                             strconv::ExpDec, false, false)
}

///
/// Convert a string in base 16 to a float.
/// Accepts a optional binary exponent.
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
/// `none` if the string did not represent a valid number.  Otherwise,
/// `Some(n)` where `n` is the floating-point number represented by `[num]`.
///
#[inline(always)]
pub fn from_str_hex(num: &str) -> Option<f64> {
    strconv::from_str_common(num, 16u, true, true, true,
                             strconv::ExpBin, false, false)
}

///
/// Convert a string in an given base to a float.
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
/// `none` if the string did not represent a valid number. Otherwise,
/// `Some(n)` where `n` is the floating-point number represented by `num`.
///
#[inline(always)]
pub fn from_str_radix(num: &str, rdx: uint) -> Option<f64> {
    strconv::from_str_common(num, rdx, true, true, false,
                             strconv::ExpNone, false, false)
}

impl from_str::FromStr for f64 {
    #[inline(always)]
    fn from_str(val: &str) -> Option<f64> { from_str(val) }
}

impl num::FromStrRadix for f64 {
    #[inline(always)]
    fn from_str_radix(val: &str, rdx: uint) -> Option<f64> {
        from_str_radix(val, rdx)
    }
}

#[cfg(test)]
mod tests {
    use f64::*;
    use super::*;
    use prelude::*;

    macro_rules! assert_fuzzy_eq(
        ($a:expr, $b:expr) => ({
            let a = $a, b = $b;
            if !((a - b).abs() < 1.0e-6) {
                fail!(fmt!("The values were not approximately equal. \
                            Found: %? and expected %?", a, b));
            }
        })
    )

    #[test]
    fn test_num() {
        num::test_num(10f64, 2f64);
    }

    #[test]
    fn test_min() {
        assert_eq!(1f64.min(&2f64), 1f64);
        assert_eq!(2f64.min(&1f64), 1f64);
        assert!(1f64.min(&Float::NaN::<f64>()).is_NaN());
        assert!(Float::NaN::<f64>().min(&1f64).is_NaN());
    }

    #[test]
    fn test_max() {
        assert_eq!(1f64.max(&2f64), 2f64);
        assert_eq!(2f64.max(&1f64), 2f64);
        assert!(1f64.max(&Float::NaN::<f64>()).is_NaN());
        assert!(Float::NaN::<f64>().max(&1f64).is_NaN());
    }

    #[test]
    fn test_clamp() {
        assert_eq!(1f64.clamp(&2f64, &4f64), 2f64);
        assert_eq!(8f64.clamp(&2f64, &4f64), 4f64);
        assert_eq!(3f64.clamp(&2f64, &4f64), 3f64);
        assert!(3f64.clamp(&Float::NaN::<f64>(), &4f64).is_NaN());
        assert!(3f64.clamp(&2f64, &Float::NaN::<f64>()).is_NaN());
        assert!(Float::NaN::<f64>().clamp(&2f64, &4f64).is_NaN());
    }

    #[test]
    fn test_floor() {
        assert_fuzzy_eq!(1.0f64.floor(), 1.0f64);
        assert_fuzzy_eq!(1.3f64.floor(), 1.0f64);
        assert_fuzzy_eq!(1.5f64.floor(), 1.0f64);
        assert_fuzzy_eq!(1.7f64.floor(), 1.0f64);
        assert_fuzzy_eq!(0.0f64.floor(), 0.0f64);
        assert_fuzzy_eq!((-0.0f64).floor(), -0.0f64);
        assert_fuzzy_eq!((-1.0f64).floor(), -1.0f64);
        assert_fuzzy_eq!((-1.3f64).floor(), -2.0f64);
        assert_fuzzy_eq!((-1.5f64).floor(), -2.0f64);
        assert_fuzzy_eq!((-1.7f64).floor(), -2.0f64);
    }

    #[test]
    fn test_ceil() {
        assert_fuzzy_eq!(1.0f64.ceil(), 1.0f64);
        assert_fuzzy_eq!(1.3f64.ceil(), 2.0f64);
        assert_fuzzy_eq!(1.5f64.ceil(), 2.0f64);
        assert_fuzzy_eq!(1.7f64.ceil(), 2.0f64);
        assert_fuzzy_eq!(0.0f64.ceil(), 0.0f64);
        assert_fuzzy_eq!((-0.0f64).ceil(), -0.0f64);
        assert_fuzzy_eq!((-1.0f64).ceil(), -1.0f64);
        assert_fuzzy_eq!((-1.3f64).ceil(), -1.0f64);
        assert_fuzzy_eq!((-1.5f64).ceil(), -1.0f64);
        assert_fuzzy_eq!((-1.7f64).ceil(), -1.0f64);
    }

    #[test]
    fn test_round() {
        assert_fuzzy_eq!(1.0f64.round(), 1.0f64);
        assert_fuzzy_eq!(1.3f64.round(), 1.0f64);
        assert_fuzzy_eq!(1.5f64.round(), 2.0f64);
        assert_fuzzy_eq!(1.7f64.round(), 2.0f64);
        assert_fuzzy_eq!(0.0f64.round(), 0.0f64);
        assert_fuzzy_eq!((-0.0f64).round(), -0.0f64);
        assert_fuzzy_eq!((-1.0f64).round(), -1.0f64);
        assert_fuzzy_eq!((-1.3f64).round(), -1.0f64);
        assert_fuzzy_eq!((-1.5f64).round(), -2.0f64);
        assert_fuzzy_eq!((-1.7f64).round(), -2.0f64);
    }

    #[test]
    fn test_trunc() {
        assert_fuzzy_eq!(1.0f64.trunc(), 1.0f64);
        assert_fuzzy_eq!(1.3f64.trunc(), 1.0f64);
        assert_fuzzy_eq!(1.5f64.trunc(), 1.0f64);
        assert_fuzzy_eq!(1.7f64.trunc(), 1.0f64);
        assert_fuzzy_eq!(0.0f64.trunc(), 0.0f64);
        assert_fuzzy_eq!((-0.0f64).trunc(), -0.0f64);
        assert_fuzzy_eq!((-1.0f64).trunc(), -1.0f64);
        assert_fuzzy_eq!((-1.3f64).trunc(), -1.0f64);
        assert_fuzzy_eq!((-1.5f64).trunc(), -1.0f64);
        assert_fuzzy_eq!((-1.7f64).trunc(), -1.0f64);
    }

    #[test]
    fn test_fract() {
        assert_fuzzy_eq!(1.0f64.fract(), 0.0f64);
        assert_fuzzy_eq!(1.3f64.fract(), 0.3f64);
        assert_fuzzy_eq!(1.5f64.fract(), 0.5f64);
        assert_fuzzy_eq!(1.7f64.fract(), 0.7f64);
        assert_fuzzy_eq!(0.0f64.fract(), 0.0f64);
        assert_fuzzy_eq!((-0.0f64).fract(), -0.0f64);
        assert_fuzzy_eq!((-1.0f64).fract(), -0.0f64);
        assert_fuzzy_eq!((-1.3f64).fract(), -0.3f64);
        assert_fuzzy_eq!((-1.5f64).fract(), -0.5f64);
        assert_fuzzy_eq!((-1.7f64).fract(), -0.7f64);
    }

    #[test]
    fn test_real_consts() {
        assert_fuzzy_eq!(Real::two_pi::<f64>(), 2.0 * Real::pi::<f64>());
        assert_fuzzy_eq!(Real::frac_pi_2::<f64>(), Real::pi::<f64>() / 2f64);
        assert_fuzzy_eq!(Real::frac_pi_3::<f64>(), Real::pi::<f64>() / 3f64);
        assert_fuzzy_eq!(Real::frac_pi_4::<f64>(), Real::pi::<f64>() / 4f64);
        assert_fuzzy_eq!(Real::frac_pi_6::<f64>(), Real::pi::<f64>() / 6f64);
        assert_fuzzy_eq!(Real::frac_pi_8::<f64>(), Real::pi::<f64>() / 8f64);
        assert_fuzzy_eq!(Real::frac_1_pi::<f64>(), 1f64 / Real::pi::<f64>());
        assert_fuzzy_eq!(Real::frac_2_pi::<f64>(), 2f64 / Real::pi::<f64>());
        assert_fuzzy_eq!(Real::frac_2_sqrtpi::<f64>(), 2f64 / Real::pi::<f64>().sqrt());
        assert_fuzzy_eq!(Real::sqrt2::<f64>(), 2f64.sqrt());
        assert_fuzzy_eq!(Real::frac_1_sqrt2::<f64>(), 1f64 / 2f64.sqrt());
        assert_fuzzy_eq!(Real::log2_e::<f64>(), Real::e::<f64>().log2());
        assert_fuzzy_eq!(Real::log10_e::<f64>(), Real::e::<f64>().log10());
        assert_fuzzy_eq!(Real::log_2::<f64>(), 2f64.log());
        assert_fuzzy_eq!(Real::log_10::<f64>(), 10f64.log());
    }

    #[test]
    pub fn test_signed() {
        assert_eq!(infinity.abs(), infinity);
        assert_eq!(1f64.abs(), 1f64);
        assert_eq!(0f64.abs(), 0f64);
        assert_eq!((-0f64).abs(), 0f64);
        assert_eq!((-1f64).abs(), 1f64);
        assert_eq!(neg_infinity.abs(), infinity);
        assert_eq!((1f64/neg_infinity).abs(), 0f64);
        assert!(NaN.abs().is_NaN());

        assert_eq!(infinity.signum(), 1f64);
        assert_eq!(1f64.signum(), 1f64);
        assert_eq!(0f64.signum(), 1f64);
        assert_eq!((-0f64).signum(), -1f64);
        assert_eq!((-1f64).signum(), -1f64);
        assert_eq!(neg_infinity.signum(), -1f64);
        assert_eq!((1f64/neg_infinity).signum(), -1f64);
        assert!(NaN.signum().is_NaN());

        assert!(infinity.is_positive());
        assert!(1f64.is_positive());
        assert!(0f64.is_positive());
        assert!(!(-0f64).is_positive());
        assert!(!(-1f64).is_positive());
        assert!(!neg_infinity.is_positive());
        assert!(!(1f64/neg_infinity).is_positive());
        assert!(!NaN.is_positive());

        assert!(!infinity.is_negative());
        assert!(!1f64.is_negative());
        assert!(!0f64.is_negative());
        assert!((-0f64).is_negative());
        assert!((-1f64).is_negative());
        assert!(neg_infinity.is_negative());
        assert!((1f64/neg_infinity).is_negative());
        assert!(!NaN.is_negative());
    }

    #[test]
    fn test_primitive() {
        assert_eq!(Primitive::bits::<f64>(), sys::size_of::<f64>() * 8);
        assert_eq!(Primitive::bytes::<f64>(), sys::size_of::<f64>());
    }
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
