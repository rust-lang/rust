// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for `f32`

use from_str;
use num::{Zero, One, strconv};
use prelude::*;

pub use cmath::c_float_targ_consts::*;

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
            use cmath::c_float_utils;
            use libc::{c_float, c_int};
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
    fn abs(n: f32) -> f32 = intrinsics::fabsf32,
    fn cos(n: f32) -> f32 = intrinsics::cosf32,
    fn exp(n: f32) -> f32 = intrinsics::expf32,
    fn exp2(n: f32) -> f32 = intrinsics::exp2f32,
    fn floor(x: f32) -> f32 = intrinsics::floorf32,
    fn ln(n: f32) -> f32 = intrinsics::logf32,
    fn log10(n: f32) -> f32 = intrinsics::log10f32,
    fn log2(n: f32) -> f32 = intrinsics::log2f32,
    fn mul_add(a: f32, b: f32, c: f32) -> f32 = intrinsics::fmaf32,
    fn pow(n: f32, e: f32) -> f32 = intrinsics::powf32,
    fn powi(n: f32, e: c_int) -> f32 = intrinsics::powif32,
    fn sin(n: f32) -> f32 = intrinsics::sinf32,
    fn sqrt(n: f32) -> f32 = intrinsics::sqrtf32,

    // LLVM 3.3 required to use intrinsics for these four
    fn ceil(n: c_float) -> c_float = c_float_utils::ceil,
    fn trunc(n: c_float) -> c_float = c_float_utils::trunc,
    /*
    fn ceil(n: f32) -> f32 = intrinsics::ceilf32,
    fn trunc(n: f32) -> f32 = intrinsics::truncf32,
    fn rint(n: f32) -> f32 = intrinsics::rintf32,
    fn nearbyint(n: f32) -> f32 = intrinsics::nearbyintf32,
    */

    // cmath
    fn acos(n: c_float) -> c_float = c_float_utils::acos,
    fn asin(n: c_float) -> c_float = c_float_utils::asin,
    fn atan(n: c_float) -> c_float = c_float_utils::atan,
    fn atan2(a: c_float, b: c_float) -> c_float = c_float_utils::atan2,
    fn cbrt(n: c_float) -> c_float = c_float_utils::cbrt,
    fn copysign(x: c_float, y: c_float) -> c_float = c_float_utils::copysign,
    fn cosh(n: c_float) -> c_float = c_float_utils::cosh,
    fn erf(n: c_float) -> c_float = c_float_utils::erf,
    fn erfc(n: c_float) -> c_float = c_float_utils::erfc,
    fn expm1(n: c_float) -> c_float = c_float_utils::expm1,
    fn abs_sub(a: c_float, b: c_float) -> c_float = c_float_utils::abs_sub,
    fn fmax(a: c_float, b: c_float) -> c_float = c_float_utils::fmax,
    fn fmin(a: c_float, b: c_float) -> c_float = c_float_utils::fmin,
    fn next_after(x: c_float, y: c_float) -> c_float = c_float_utils::next_after,
    fn frexp(n: c_float, value: &mut c_int) -> c_float = c_float_utils::frexp,
    fn hypot(x: c_float, y: c_float) -> c_float = c_float_utils::hypot,
    fn ldexp(x: c_float, n: c_int) -> c_float = c_float_utils::ldexp,
    fn lgamma(n: c_float, sign: &mut c_int) -> c_float = c_float_utils::lgamma,
    fn log_radix(n: c_float) -> c_float = c_float_utils::log_radix,
    fn ln1p(n: c_float) -> c_float = c_float_utils::ln1p,
    fn ilog_radix(n: c_float) -> c_int = c_float_utils::ilog_radix,
    fn modf(n: c_float, iptr: &mut c_float) -> c_float = c_float_utils::modf,
    fn round(n: c_float) -> c_float = c_float_utils::round,
    fn ldexp_radix(n: c_float, i: c_int) -> c_float = c_float_utils::ldexp_radix,
    fn sinh(n: c_float) -> c_float = c_float_utils::sinh,
    fn tan(n: c_float) -> c_float = c_float_utils::tan,
    fn tanh(n: c_float) -> c_float = c_float_utils::tanh,
    fn tgamma(n: c_float) -> c_float = c_float_utils::tgamma
)

// These are not defined inside consts:: for consistency with
// the integer types

pub static NaN: f32 = 0.0_f32/0.0_f32;

pub static infinity: f32 = 1.0_f32/0.0_f32;

pub static neg_infinity: f32 = -1.0_f32/0.0_f32;

#[inline(always)]
pub fn add(x: f32, y: f32) -> f32 { return x + y; }

#[inline(always)]
pub fn sub(x: f32, y: f32) -> f32 { return x - y; }

#[inline(always)]
pub fn mul(x: f32, y: f32) -> f32 { return x * y; }

#[inline(always)]
pub fn div(x: f32, y: f32) -> f32 { return x / y; }

#[inline(always)]
pub fn rem(x: f32, y: f32) -> f32 { return x % y; }

#[inline(always)]
pub fn lt(x: f32, y: f32) -> bool { return x < y; }

#[inline(always)]
pub fn le(x: f32, y: f32) -> bool { return x <= y; }

#[inline(always)]
pub fn eq(x: f32, y: f32) -> bool { return x == y; }

#[inline(always)]
pub fn ne(x: f32, y: f32) -> bool { return x != y; }

#[inline(always)]
pub fn ge(x: f32, y: f32) -> bool { return x >= y; }

#[inline(always)]
pub fn gt(x: f32, y: f32) -> bool { return x > y; }


// FIXME (#1999): replace the predicates below with llvm intrinsics or
// calls to the libmath macros in the rust runtime for performance.

// FIXME (#1999): add is_normal, is_subnormal, and fpclassify.

/* Module: consts */
pub mod consts {
    // FIXME (requires Issue #1433 to fix): replace with mathematical
    // staticants from cmath.
    /// Archimedes' staticant
    pub static pi: f32 = 3.14159265358979323846264338327950288_f32;

    /// pi/2.0
    pub static frac_pi_2: f32 = 1.57079632679489661923132169163975144_f32;

    /// pi/4.0
    pub static frac_pi_4: f32 = 0.785398163397448309615660845819875721_f32;

    /// 1.0/pi
    pub static frac_1_pi: f32 = 0.318309886183790671537767526745028724_f32;

    /// 2.0/pi
    pub static frac_2_pi: f32 = 0.636619772367581343075535053490057448_f32;

    /// 2.0/sqrt(pi)
    pub static frac_2_sqrtpi: f32 = 1.12837916709551257389615890312154517_f32;

    /// sqrt(2.0)
    pub static sqrt2: f32 = 1.41421356237309504880168872420969808_f32;

    /// 1.0/sqrt(2.0)
    pub static frac_1_sqrt2: f32 = 0.707106781186547524400844362104849039_f32;

    /// Euler's number
    pub static e: f32 = 2.71828182845904523536028747135266250_f32;

    /// log2(e)
    pub static log2_e: f32 = 1.44269504088896340735992468100189214_f32;

    /// log10(e)
    pub static log10_e: f32 = 0.434294481903251827651128918916605082_f32;

    /// ln(2.0)
    pub static ln_2: f32 = 0.693147180559945309417232121458176568_f32;

    /// ln(10.0)
    pub static ln_10: f32 = 2.30258509299404568401799145468436421_f32;
}

#[inline(always)]
pub fn logarithm(n: f32, b: f32) -> f32 {
    return log2(n) / log2(b);
}

impl Num for f32 {}

#[cfg(notest)]
impl Eq for f32 {
    #[inline(always)]
    fn eq(&self, other: &f32) -> bool { (*self) == (*other) }
    #[inline(always)]
    fn ne(&self, other: &f32) -> bool { (*self) != (*other) }
}

#[cfg(notest)]
impl Ord for f32 {
    #[inline(always)]
    fn lt(&self, other: &f32) -> bool { (*self) < (*other) }
    #[inline(always)]
    fn le(&self, other: &f32) -> bool { (*self) <= (*other) }
    #[inline(always)]
    fn ge(&self, other: &f32) -> bool { (*self) >= (*other) }
    #[inline(always)]
    fn gt(&self, other: &f32) -> bool { (*self) > (*other) }
}

impl Orderable for f32 {
    /// Returns `NaN` if either of the numbers are `NaN`.
    #[inline(always)]
    fn min(&self, other: &f32) -> f32 {
        if self.is_NaN() || other.is_NaN() { Float::NaN() } else { fmin(*self, *other) }
    }

    /// Returns `NaN` if either of the numbers are `NaN`.
    #[inline(always)]
    fn max(&self, other: &f32) -> f32 {
        if self.is_NaN() || other.is_NaN() { Float::NaN() } else { fmax(*self, *other) }
    }

    /// Returns the number constrained within the range `mn <= self <= mx`.
    /// If any of the numbers are `NaN` then `NaN` is returned.
    #[inline(always)]
    fn clamp(&self, mn: &f32, mx: &f32) -> f32 {
        if self.is_NaN() { *self }
        else if !(*self <= *mx) { *mx }
        else if !(*self >= *mn) { *mn }
        else { *self }
    }
}

impl Zero for f32 {
    #[inline(always)]
    fn zero() -> f32 { 0.0 }

    /// Returns true if the number is equal to either `0.0` or `-0.0`
    #[inline(always)]
    fn is_zero(&self) -> bool { *self == 0.0 || *self == -0.0 }
}

impl One for f32 {
    #[inline(always)]
    fn one() -> f32 { 1.0 }
}

#[cfg(notest)]
impl Add<f32,f32> for f32 {
    #[inline(always)]
    fn add(&self, other: &f32) -> f32 { *self + *other }
}

#[cfg(notest)]
impl Sub<f32,f32> for f32 {
    #[inline(always)]
    fn sub(&self, other: &f32) -> f32 { *self - *other }
}

#[cfg(notest)]
impl Mul<f32,f32> for f32 {
    #[inline(always)]
    fn mul(&self, other: &f32) -> f32 { *self * *other }
}

#[cfg(notest)]
impl Div<f32,f32> for f32 {
    #[inline(always)]
    fn div(&self, other: &f32) -> f32 { *self / *other }
}

#[cfg(stage0,notest)]
impl Modulo<f32,f32> for f32 {
    #[inline(always)]
    fn modulo(&self, other: &f32) -> f32 { *self % *other }
}
#[cfg(not(stage0),notest)]
impl Rem<f32,f32> for f32 {
    #[inline(always)]
    fn rem(&self, other: &f32) -> f32 { *self % *other }
}

#[cfg(notest)]
impl Neg<f32> for f32 {
    #[inline(always)]
    fn neg(&self) -> f32 { -*self }
}

impl Signed for f32 {
    /// Computes the absolute value. Returns `NaN` if the number is `NaN`.
    #[inline(always)]
    fn abs(&self) -> f32 { abs(*self) }

    ///
    /// # Returns
    ///
    /// - `1.0` if the number is positive, `+0.0` or `infinity`
    /// - `-1.0` if the number is negative, `-0.0` or `neg_infinity`
    /// - `NaN` if the number is NaN
    ///
    #[inline(always)]
    fn signum(&self) -> f32 {
        if self.is_NaN() { NaN } else { copysign(1.0, *self) }
    }

    /// Returns `true` if the number is positive, including `+0.0` and `infinity`
    #[inline(always)]
    fn is_positive(&self) -> bool { *self > 0.0 || (1.0 / *self) == infinity }

    /// Returns `true` if the number is negative, including `-0.0` and `neg_infinity`
    #[inline(always)]
    fn is_negative(&self) -> bool { *self < 0.0 || (1.0 / *self) == neg_infinity }
}

impl Round for f32 {
    /// Round half-way cases toward `neg_infinity`
    #[inline(always)]
    fn floor(&self) -> f32 { floor(*self) }

    /// Round half-way cases toward `infinity`
    #[inline(always)]
    fn ceil(&self) -> f32 { ceil(*self) }

    /// Round half-way cases away from `0.0`
    #[inline(always)]
    fn round(&self) -> f32 { round(*self) }

    /// The integer part of the number (rounds towards `0.0`)
    #[inline(always)]
    fn trunc(&self) -> f32 { trunc(*self) }

    ///
    /// The fractional part of the number, satisfying:
    ///
    /// ~~~
    /// assert!(x == trunc(x) + fract(x))
    /// ~~~
    ///
    #[inline(always)]
    fn fract(&self) -> f32 { *self - self.trunc() }
}

impl Fractional for f32 {
    /// The reciprocal (multiplicative inverse) of the number
    #[inline(always)]
    fn recip(&self) -> f32 { 1.0 / *self }
}

impl Algebraic for f32 {
    #[inline(always)]
    fn pow(&self, n: f32) -> f32 { pow(*self, n) }

    #[inline(always)]
    fn sqrt(&self) -> f32 { sqrt(*self) }

    #[inline(always)]
    fn rsqrt(&self) -> f32 { self.sqrt().recip() }

    #[inline(always)]
    fn cbrt(&self) -> f32 { cbrt(*self) }

    #[inline(always)]
    fn hypot(&self, other: f32) -> f32 { hypot(*self, other) }
}

impl Trigonometric for f32 {
    #[inline(always)]
    fn sin(&self) -> f32 { sin(*self) }

    #[inline(always)]
    fn cos(&self) -> f32 { cos(*self) }

    #[inline(always)]
    fn tan(&self) -> f32 { tan(*self) }

    #[inline(always)]
    fn asin(&self) -> f32 { asin(*self) }

    #[inline(always)]
    fn acos(&self) -> f32 { acos(*self) }

    #[inline(always)]
    fn atan(&self) -> f32 { atan(*self) }

    #[inline(always)]
    fn atan2(&self, other: f32) -> f32 { atan2(*self, other) }
}

impl Exponential for f32 {
    #[inline(always)]
    fn exp(&self) -> f32 { exp(*self) }

    #[inline(always)]
    fn exp2(&self) -> f32 { exp2(*self) }

    #[inline(always)]
    fn expm1(&self) -> f32 { expm1(*self) }

    #[inline(always)]
    fn log(&self) -> f32 { ln(*self) }

    #[inline(always)]
    fn log2(&self) -> f32 { log2(*self) }

    #[inline(always)]
    fn log10(&self) -> f32 { log10(*self) }
}

impl Hyperbolic for f32 {
    #[inline(always)]
    fn sinh(&self) -> f32 { sinh(*self) }

    #[inline(always)]
    fn cosh(&self) -> f32 { cosh(*self) }

    #[inline(always)]
    fn tanh(&self) -> f32 { tanh(*self) }
}

impl Real for f32 {
    /// Archimedes' constant
    #[inline(always)]
    fn pi() -> f32 { 3.14159265358979323846264338327950288 }

    /// 2.0 * pi
    #[inline(always)]
    fn two_pi() -> f32 { 6.28318530717958647692528676655900576 }

    /// pi / 2.0
    #[inline(always)]
    fn frac_pi_2() -> f32 { 1.57079632679489661923132169163975144 }

    /// pi / 3.0
    #[inline(always)]
    fn frac_pi_3() -> f32 { 1.04719755119659774615421446109316763 }

    /// pi / 4.0
    #[inline(always)]
    fn frac_pi_4() -> f32 { 0.785398163397448309615660845819875721 }

    /// pi / 6.0
    #[inline(always)]
    fn frac_pi_6() -> f32 { 0.52359877559829887307710723054658381 }

    /// pi / 8.0
    #[inline(always)]
    fn frac_pi_8() -> f32 { 0.39269908169872415480783042290993786 }

    /// 1 .0/ pi
    #[inline(always)]
    fn frac_1_pi() -> f32 { 0.318309886183790671537767526745028724 }

    /// 2.0 / pi
    #[inline(always)]
    fn frac_2_pi() -> f32 { 0.636619772367581343075535053490057448 }

    /// 2.0 / sqrt(pi)
    #[inline(always)]
    fn frac_2_sqrtpi() -> f32 { 1.12837916709551257389615890312154517 }

    /// sqrt(2.0)
    #[inline(always)]
    fn sqrt2() -> f32 { 1.41421356237309504880168872420969808 }

    /// 1.0 / sqrt(2.0)
    #[inline(always)]
    fn frac_1_sqrt2() -> f32 { 0.707106781186547524400844362104849039 }

    /// Euler's number
    #[inline(always)]
    fn e() -> f32 { 2.71828182845904523536028747135266250 }

    /// log2(e)
    #[inline(always)]
    fn log2_e() -> f32 { 1.44269504088896340735992468100189214 }

    /// log10(e)
    #[inline(always)]
    fn log10_e() -> f32 { 0.434294481903251827651128918916605082 }

    /// log(2.0)
    #[inline(always)]
    fn log_2() -> f32 { 0.693147180559945309417232121458176568 }

    /// log(10.0)
    #[inline(always)]
    fn log_10() -> f32 { 2.30258509299404568401799145468436421 }

    /// Converts to degrees, assuming the number is in radians
    #[inline(always)]
    fn to_degrees(&self) -> f32 { *self * (180.0 / Real::pi::<f32>()) }

    /// Converts to radians, assuming the number is in degrees
    #[inline(always)]
    fn to_radians(&self) -> f32 { *self * (Real::pi::<f32>() / 180.0) }
}

impl Bounded for f32 {
    #[inline(always)]
    fn min_value() -> f32 { 1.17549435e-38 }

    #[inline(always)]
    fn max_value() -> f32 { 3.40282347e+38 }
}

impl Primitive for f32 {
    #[inline(always)]
    fn bits() -> uint { 32 }

    #[inline(always)]
    fn bytes() -> uint { Primitive::bits::<f32>() / 8 }
}

impl Float for f32 {
    #[inline(always)]
    fn NaN() -> f32 { 0.0 / 0.0 }

    #[inline(always)]
    fn infinity() -> f32 { 1.0 / 0.0 }

    #[inline(always)]
    fn neg_infinity() -> f32 { -1.0 / 0.0 }

    #[inline(always)]
    fn neg_zero() -> f32 { -0.0 }

    #[inline(always)]
    fn is_NaN(&self) -> bool { *self != *self }

    #[inline(always)]
    fn mantissa_digits() -> uint { 24 }

    #[inline(always)]
    fn digits() -> uint { 6 }

    #[inline(always)]
    fn epsilon() -> f32 { 1.19209290e-07 }

    #[inline(always)]
    fn min_exp() -> int { -125 }

    #[inline(always)]
    fn max_exp() -> int { 128 }

    #[inline(always)]
    fn min_10_exp() -> int { -37 }

    #[inline(always)]
    fn max_10_exp() -> int { 38 }

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

    ///
    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding error. This
    /// produces a more accurate result with better performance than a separate multiplication
    /// operation followed by an add.
    ///
    #[inline(always)]
    fn mul_add(&self, a: f32, b: f32) -> f32 {
        mul_add(*self, a, b)
    }

    /// Returns the next representable floating-point value in the direction of `other`
    #[inline(always)]
    fn next_after(&self, other: f32) -> f32 {
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
pub fn to_str(num: f32) -> ~str {
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
pub fn to_str_hex(num: f32) -> ~str {
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
pub fn to_str_radix(num: f32, rdx: uint) -> ~str {
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
pub fn to_str_radix_special(num: f32, rdx: uint) -> (~str, bool) {
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
pub fn to_str_exact(num: f32, dig: uint) -> ~str {
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
pub fn to_str_digits(num: f32, dig: uint) -> ~str {
    let (r, _) = strconv::to_str_common(
        &num, 10u, true, strconv::SignNeg, strconv::DigMax(dig));
    r
}

impl to_str::ToStr for f32 {
    #[inline(always)]
    fn to_str(&self) -> ~str { to_str_digits(*self, 8) }
}

impl num::ToStrRadix for f32 {
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
pub fn from_str(num: &str) -> Option<f32> {
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
pub fn from_str_hex(num: &str) -> Option<f32> {
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
pub fn from_str_radix(num: &str, rdx: uint) -> Option<f32> {
    strconv::from_str_common(num, rdx, true, true, false,
                             strconv::ExpNone, false, false)
}

impl from_str::FromStr for f32 {
    #[inline(always)]
    fn from_str(val: &str) -> Option<f32> { from_str(val) }
}

impl num::FromStrRadix for f32 {
    #[inline(always)]
    fn from_str_radix(val: &str, rdx: uint) -> Option<f32> {
        from_str_radix(val, rdx)
    }
}

#[cfg(test)]
mod tests {
    use f32::*;
    use super::*;
    use prelude::*;

    macro_rules! assert_fuzzy_eq(
        ($a:expr, $b:expr) => ({
            let a = $a, b = $b;
            if !((a - b).abs() < 1.0e-6) {
                fail!(fmt!("The values were not approximately equal. Found: %? and %?", a, b));
            }
        })
    )

    #[test]
    fn test_num() {
        num::test_num(10f32, 2f32);
    }

    #[test]
    fn test_min() {
        assert_eq!(1f32.min(&2f32), 1f32);
        assert_eq!(2f32.min(&1f32), 1f32);
    }

    #[test]
    fn test_max() {
        assert_eq!(1f32.max(&2f32), 2f32);
        assert_eq!(2f32.max(&1f32), 2f32);
    }

    #[test]
    fn test_clamp() {
        assert_eq!(1f32.clamp(&2f32, &4f32), 2f32);
        assert_eq!(8f32.clamp(&2f32, &4f32), 4f32);
        assert_eq!(3f32.clamp(&2f32, &4f32), 3f32);
        assert!(3f32.clamp(&Float::NaN::<f32>(), &4f32).is_NaN());
        assert!(3f32.clamp(&2f32, &Float::NaN::<f32>()).is_NaN());
        assert!(Float::NaN::<f32>().clamp(&2f32, &4f32).is_NaN());
    }

    #[test]
    fn test_floor() {
        assert_fuzzy_eq!(1.0f32.floor(), 1.0f32);
        assert_fuzzy_eq!(1.3f32.floor(), 1.0f32);
        assert_fuzzy_eq!(1.5f32.floor(), 1.0f32);
        assert_fuzzy_eq!(1.7f32.floor(), 1.0f32);
        assert_fuzzy_eq!(0.0f32.floor(), 0.0f32);
        assert_fuzzy_eq!((-0.0f32).floor(), -0.0f32);
        assert_fuzzy_eq!((-1.0f32).floor(), -1.0f32);
        assert_fuzzy_eq!((-1.3f32).floor(), -2.0f32);
        assert_fuzzy_eq!((-1.5f32).floor(), -2.0f32);
        assert_fuzzy_eq!((-1.7f32).floor(), -2.0f32);
    }

    #[test]
    fn test_ceil() {
        assert_fuzzy_eq!(1.0f32.ceil(), 1.0f32);
        assert_fuzzy_eq!(1.3f32.ceil(), 2.0f32);
        assert_fuzzy_eq!(1.5f32.ceil(), 2.0f32);
        assert_fuzzy_eq!(1.7f32.ceil(), 2.0f32);
        assert_fuzzy_eq!(0.0f32.ceil(), 0.0f32);
        assert_fuzzy_eq!((-0.0f32).ceil(), -0.0f32);
        assert_fuzzy_eq!((-1.0f32).ceil(), -1.0f32);
        assert_fuzzy_eq!((-1.3f32).ceil(), -1.0f32);
        assert_fuzzy_eq!((-1.5f32).ceil(), -1.0f32);
        assert_fuzzy_eq!((-1.7f32).ceil(), -1.0f32);
    }

    #[test]
    fn test_round() {
        assert_fuzzy_eq!(1.0f32.round(), 1.0f32);
        assert_fuzzy_eq!(1.3f32.round(), 1.0f32);
        assert_fuzzy_eq!(1.5f32.round(), 2.0f32);
        assert_fuzzy_eq!(1.7f32.round(), 2.0f32);
        assert_fuzzy_eq!(0.0f32.round(), 0.0f32);
        assert_fuzzy_eq!((-0.0f32).round(), -0.0f32);
        assert_fuzzy_eq!((-1.0f32).round(), -1.0f32);
        assert_fuzzy_eq!((-1.3f32).round(), -1.0f32);
        assert_fuzzy_eq!((-1.5f32).round(), -2.0f32);
        assert_fuzzy_eq!((-1.7f32).round(), -2.0f32);
    }

    #[test]
    fn test_trunc() {
        assert_fuzzy_eq!(1.0f32.trunc(), 1.0f32);
        assert_fuzzy_eq!(1.3f32.trunc(), 1.0f32);
        assert_fuzzy_eq!(1.5f32.trunc(), 1.0f32);
        assert_fuzzy_eq!(1.7f32.trunc(), 1.0f32);
        assert_fuzzy_eq!(0.0f32.trunc(), 0.0f32);
        assert_fuzzy_eq!((-0.0f32).trunc(), -0.0f32);
        assert_fuzzy_eq!((-1.0f32).trunc(), -1.0f32);
        assert_fuzzy_eq!((-1.3f32).trunc(), -1.0f32);
        assert_fuzzy_eq!((-1.5f32).trunc(), -1.0f32);
        assert_fuzzy_eq!((-1.7f32).trunc(), -1.0f32);
    }

    #[test]
    fn test_fract() {
        assert_fuzzy_eq!(1.0f32.fract(), 0.0f32);
        assert_fuzzy_eq!(1.3f32.fract(), 0.3f32);
        assert_fuzzy_eq!(1.5f32.fract(), 0.5f32);
        assert_fuzzy_eq!(1.7f32.fract(), 0.7f32);
        assert_fuzzy_eq!(0.0f32.fract(), 0.0f32);
        assert_fuzzy_eq!((-0.0f32).fract(), -0.0f32);
        assert_fuzzy_eq!((-1.0f32).fract(), -0.0f32);
        assert_fuzzy_eq!((-1.3f32).fract(), -0.3f32);
        assert_fuzzy_eq!((-1.5f32).fract(), -0.5f32);
        assert_fuzzy_eq!((-1.7f32).fract(), -0.7f32);
    }

    #[test]
    fn test_real_consts() {
        assert_fuzzy_eq!(Real::two_pi::<f32>(), 2f32 * Real::pi::<f32>());
        assert_fuzzy_eq!(Real::frac_pi_2::<f32>(), Real::pi::<f32>() / 2f32);
        assert_fuzzy_eq!(Real::frac_pi_3::<f32>(), Real::pi::<f32>() / 3f32);
        assert_fuzzy_eq!(Real::frac_pi_4::<f32>(), Real::pi::<f32>() / 4f32);
        assert_fuzzy_eq!(Real::frac_pi_6::<f32>(), Real::pi::<f32>() / 6f32);
        assert_fuzzy_eq!(Real::frac_pi_8::<f32>(), Real::pi::<f32>() / 8f32);
        assert_fuzzy_eq!(Real::frac_1_pi::<f32>(), 1f32 / Real::pi::<f32>());
        assert_fuzzy_eq!(Real::frac_2_pi::<f32>(), 2f32 / Real::pi::<f32>());
        assert_fuzzy_eq!(Real::frac_2_sqrtpi::<f32>(), 2f32 / Real::pi::<f32>().sqrt());
        assert_fuzzy_eq!(Real::sqrt2::<f32>(), 2f32.sqrt());
        assert_fuzzy_eq!(Real::frac_1_sqrt2::<f32>(), 1f32 / 2f32.sqrt());
        assert_fuzzy_eq!(Real::log2_e::<f32>(), Real::e::<f32>().log2());
        assert_fuzzy_eq!(Real::log10_e::<f32>(), Real::e::<f32>().log10());
        assert_fuzzy_eq!(Real::log_2::<f32>(), 2f32.log());
        assert_fuzzy_eq!(Real::log_10::<f32>(), 10f32.log());
    }

    #[test]
    pub fn test_signed() {
        assert_eq!(infinity.abs(), infinity);
        assert_eq!(1f32.abs(), 1f32);
        assert_eq!(0f32.abs(), 0f32);
        assert_eq!((-0f32).abs(), 0f32);
        assert_eq!((-1f32).abs(), 1f32);
        assert_eq!(neg_infinity.abs(), infinity);
        assert_eq!((1f32/neg_infinity).abs(), 0f32);
        assert!(NaN.abs().is_NaN());

        assert_eq!(infinity.signum(), 1f32);
        assert_eq!(1f32.signum(), 1f32);
        assert_eq!(0f32.signum(), 1f32);
        assert_eq!((-0f32).signum(), -1f32);
        assert_eq!((-1f32).signum(), -1f32);
        assert_eq!(neg_infinity.signum(), -1f32);
        assert_eq!((1f32/neg_infinity).signum(), -1f32);
        assert!(NaN.signum().is_NaN());

        assert!(infinity.is_positive());
        assert!(1f32.is_positive());
        assert!(0f32.is_positive());
        assert!(!(-0f32).is_positive());
        assert!(!(-1f32).is_positive());
        assert!(!neg_infinity.is_positive());
        assert!(!(1f32/neg_infinity).is_positive());
        assert!(!NaN.is_positive());

        assert!(!infinity.is_negative());
        assert!(!1f32.is_negative());
        assert!(!0f32.is_negative());
        assert!((-0f32).is_negative());
        assert!((-1f32).is_negative());
        assert!(neg_infinity.is_negative());
        assert!((1f32/neg_infinity).is_negative());
        assert!(!NaN.is_negative());
    }

    #[test]
    fn test_primitive() {
        assert_eq!(Primitive::bits::<f32>(), sys::size_of::<f32>() * 8);
        assert_eq!(Primitive::bytes::<f32>(), sys::size_of::<f32>());
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
