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

use cmath;
use libc::{c_double, c_int};
use num::strconv;
use num;
use option::Option;
use unstable::intrinsics::floorf64;
use to_str;
use from_str;

#[cfg(notest)] use cmp;
#[cfg(notest)] use ops;

pub use cmath::c_double_targ_consts::*;
pub use cmp::{min, max};

macro_rules! delegate(
    (
        fn $name:ident(
            $(
                $arg:ident : $arg_ty:ty
            ),*
        ) -> $rv:ty = $bound_name:path
    ) => (
        pub fn $name($( $arg : $arg_ty ),*) -> $rv {
            unsafe {
                $bound_name($( $arg ),*)
            }
        }
    )
)

delegate!(fn acos(n: c_double) -> c_double = cmath::c_double_utils::acos)
delegate!(fn asin(n: c_double) -> c_double = cmath::c_double_utils::asin)
delegate!(fn atan(n: c_double) -> c_double = cmath::c_double_utils::atan)
delegate!(fn atan2(a: c_double, b: c_double) -> c_double =
    cmath::c_double_utils::atan2)
delegate!(fn cbrt(n: c_double) -> c_double = cmath::c_double_utils::cbrt)
delegate!(fn ceil(n: c_double) -> c_double = cmath::c_double_utils::ceil)
delegate!(fn copysign(x: c_double, y: c_double) -> c_double =
    cmath::c_double_utils::copysign)
delegate!(fn cos(n: c_double) -> c_double = cmath::c_double_utils::cos)
delegate!(fn cosh(n: c_double) -> c_double = cmath::c_double_utils::cosh)
delegate!(fn erf(n: c_double) -> c_double = cmath::c_double_utils::erf)
delegate!(fn erfc(n: c_double) -> c_double = cmath::c_double_utils::erfc)
delegate!(fn exp(n: c_double) -> c_double = cmath::c_double_utils::exp)
delegate!(fn expm1(n: c_double) -> c_double = cmath::c_double_utils::expm1)
delegate!(fn exp2(n: c_double) -> c_double = cmath::c_double_utils::exp2)
delegate!(fn abs(n: c_double) -> c_double = cmath::c_double_utils::abs)
delegate!(fn abs_sub(a: c_double, b: c_double) -> c_double =
    cmath::c_double_utils::abs_sub)
delegate!(fn mul_add(a: c_double, b: c_double, c: c_double) -> c_double =
    cmath::c_double_utils::mul_add)
delegate!(fn fmax(a: c_double, b: c_double) -> c_double =
    cmath::c_double_utils::fmax)
delegate!(fn fmin(a: c_double, b: c_double) -> c_double =
    cmath::c_double_utils::fmin)
delegate!(fn nextafter(x: c_double, y: c_double) -> c_double =
    cmath::c_double_utils::nextafter)
delegate!(fn frexp(n: c_double, value: &mut c_int) -> c_double =
    cmath::c_double_utils::frexp)
delegate!(fn hypot(x: c_double, y: c_double) -> c_double =
    cmath::c_double_utils::hypot)
delegate!(fn ldexp(x: c_double, n: c_int) -> c_double =
    cmath::c_double_utils::ldexp)
delegate!(fn lgamma(n: c_double, sign: &mut c_int) -> c_double =
    cmath::c_double_utils::lgamma)
delegate!(fn ln(n: c_double) -> c_double = cmath::c_double_utils::ln)
delegate!(fn log_radix(n: c_double) -> c_double =
    cmath::c_double_utils::log_radix)
delegate!(fn ln1p(n: c_double) -> c_double = cmath::c_double_utils::ln1p)
delegate!(fn log10(n: c_double) -> c_double = cmath::c_double_utils::log10)
delegate!(fn log2(n: c_double) -> c_double = cmath::c_double_utils::log2)
delegate!(fn ilog_radix(n: c_double) -> c_int =
    cmath::c_double_utils::ilog_radix)
delegate!(fn modf(n: c_double, iptr: &mut c_double) -> c_double =
    cmath::c_double_utils::modf)
delegate!(fn pow(n: c_double, e: c_double) -> c_double =
    cmath::c_double_utils::pow)
delegate!(fn round(n: c_double) -> c_double = cmath::c_double_utils::round)
delegate!(fn ldexp_radix(n: c_double, i: c_int) -> c_double =
    cmath::c_double_utils::ldexp_radix)
delegate!(fn sin(n: c_double) -> c_double = cmath::c_double_utils::sin)
delegate!(fn sinh(n: c_double) -> c_double = cmath::c_double_utils::sinh)
delegate!(fn sqrt(n: c_double) -> c_double = cmath::c_double_utils::sqrt)
delegate!(fn tan(n: c_double) -> c_double = cmath::c_double_utils::tan)
delegate!(fn tanh(n: c_double) -> c_double = cmath::c_double_utils::tanh)
delegate!(fn tgamma(n: c_double) -> c_double = cmath::c_double_utils::tgamma)
delegate!(fn trunc(n: c_double) -> c_double = cmath::c_double_utils::trunc)
delegate!(fn j0(n: c_double) -> c_double = cmath::c_double_utils::j0)
delegate!(fn j1(n: c_double) -> c_double = cmath::c_double_utils::j1)
delegate!(fn jn(i: c_int, n: c_double) -> c_double =
    cmath::c_double_utils::jn)
delegate!(fn y0(n: c_double) -> c_double = cmath::c_double_utils::y0)
delegate!(fn y1(n: c_double) -> c_double = cmath::c_double_utils::y1)
delegate!(fn yn(i: c_int, n: c_double) -> c_double =
    cmath::c_double_utils::yn)

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
pub fn is_NaN(f: f64) -> bool { f != f }

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

/// Returns true if `x` is a positive number, including +0.0f640 and +Infinity
#[inline(always)]
pub fn is_positive(x: f64) -> bool
    { return x > 0.0f64 || (1.0f64/x) == infinity; }

/// Returns true if `x` is a negative number, including -0.0f640 and -Infinity
#[inline(always)]
pub fn is_negative(x: f64) -> bool
    { return x < 0.0f64 || (1.0f64/x) == neg_infinity; }

/**
 * Returns true if `x` is a negative number, including -0.0f640 and -Infinity
 *
 * This is the same as `f64::is_negative`.
 */
#[inline(always)]
pub fn is_nonpositive(x: f64) -> bool {
  return x < 0.0f64 || (1.0f64/x) == neg_infinity;
}

/**
 * Returns true if `x` is a positive number, including +0.0f640 and +Infinity
 *
 * This is the same as `f64::positive`.
 */
#[inline(always)]
pub fn is_nonnegative(x: f64) -> bool {
  return x > 0.0f64 || (1.0f64/x) == infinity;
}

/// Returns true if `x` is a zero number (positive or negative zero)
#[inline(always)]
pub fn is_zero(x: f64) -> bool {
    return x == 0.0f64 || x == -0.0f64;
}

/// Returns true if `x`is an infinite number
#[inline(always)]
pub fn is_infinite(x: f64) -> bool {
    return x == infinity || x == neg_infinity;
}

/// Returns true if `x` is a finite number
#[inline(always)]
pub fn is_finite(x: f64) -> bool {
    return !(is_NaN(x) || is_infinite(x));
}

/// Returns `x` rounded down
#[inline(always)]
pub fn floor(x: f64) -> f64 { unsafe { floorf64(x) } }

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
pub fn signbit(x: f64) -> int {
    if is_negative(x) { return 1; } else { return 0; }
}

#[inline(always)]
pub fn logarithm(n: f64, b: f64) -> f64 {
    return log2(n) / log2(b);
}

#[cfg(notest)]
impl cmp::Eq for f64 {
    #[inline(always)]
    fn eq(&self, other: &f64) -> bool { (*self) == (*other) }
    #[inline(always)]
    fn ne(&self, other: &f64) -> bool { (*self) != (*other) }
}

#[cfg(notest)]
impl cmp::Ord for f64 {
    #[inline(always)]
    fn lt(&self, other: &f64) -> bool { (*self) < (*other) }
    #[inline(always)]
    fn le(&self, other: &f64) -> bool { (*self) <= (*other) }
    #[inline(always)]
    fn ge(&self, other: &f64) -> bool { (*self) >= (*other) }
    #[inline(always)]
    fn gt(&self, other: &f64) -> bool { (*self) > (*other) }
}

impl num::Zero for f64 {
    #[inline(always)]
    fn zero() -> f64 { 0.0 }
}

impl num::One for f64 {
    #[inline(always)]
    fn one() -> f64 { 1.0 }
}

#[cfg(notest)]
impl ops::Add<f64,f64> for f64 {
    fn add(&self, other: &f64) -> f64 { *self + *other }
}
#[cfg(notest)]
impl ops::Sub<f64,f64> for f64 {
    fn sub(&self, other: &f64) -> f64 { *self - *other }
}
#[cfg(notest)]
impl ops::Mul<f64,f64> for f64 {
    fn mul(&self, other: &f64) -> f64 { *self * *other }
}
#[cfg(notest)]
impl ops::Div<f64,f64> for f64 {
    fn div(&self, other: &f64) -> f64 { *self / *other }
}
#[cfg(notest)]
impl ops::Modulo<f64,f64> for f64 {
    fn modulo(&self, other: &f64) -> f64 { *self % *other }
}
#[cfg(notest)]
impl ops::Neg<f64> for f64 {
    fn neg(&self) -> f64 { -*self }
}

impl num::Round for f64 {
    #[inline(always)]
    fn round(&self, mode: num::RoundMode) -> f64 {
        match mode {
            num::RoundDown                           => floor(*self),
            num::RoundUp                             => ceil(*self),
            num::RoundToZero   if is_negative(*self) => ceil(*self),
            num::RoundToZero                         => floor(*self),
            num::RoundFromZero if is_negative(*self) => floor(*self),
            num::RoundFromZero                       => ceil(*self)
        }
    }

    #[inline(always)]
    fn floor(&self) -> f64 { floor(*self) }
    #[inline(always)]
    fn ceil(&self) -> f64 { ceil(*self) }
    #[inline(always)]
    fn fract(&self) -> f64 {
        if is_negative(*self) {
            (*self) - ceil(*self)
        } else {
            (*self) - floor(*self)
        }
    }
}

/**
 * Section: String Conversions
 */

/**
 * Converts a float to a string
 *
 * # Arguments
 *
 * * num - The float value
 */
#[inline(always)]
pub fn to_str(num: f64) -> ~str {
    let (r, _) = strconv::to_str_common(
        &num, 10u, true, strconv::SignNeg, strconv::DigAll);
    r
}

/**
 * Converts a float to a string in hexadecimal format
 *
 * # Arguments
 *
 * * num - The float value
 */
#[inline(always)]
pub fn to_str_hex(num: f64) -> ~str {
    let (r, _) = strconv::to_str_common(
        &num, 16u, true, strconv::SignNeg, strconv::DigAll);
    r
}

/**
 * Converts a float to a string in a given radix
 *
 * # Arguments
 *
 * * num - The float value
 * * radix - The base to use
 *
 * # Failure
 *
 * Fails if called on a special value like `inf`, `-inf` or `NaN` due to
 * possible misinterpretation of the result at higher bases. If those values
 * are expected, use `to_str_radix_special()` instead.
 */
#[inline(always)]
pub fn to_str_radix(num: f64, rdx: uint) -> ~str {
    let (r, special) = strconv::to_str_common(
        &num, rdx, true, strconv::SignNeg, strconv::DigAll);
    if special { fail!(~"number has a special value, \
                      try to_str_radix_special() if those are expected") }
    r
}

/**
 * Converts a float to a string in a given radix, and a flag indicating
 * whether it's a special value
 *
 * # Arguments
 *
 * * num - The float value
 * * radix - The base to use
 */
#[inline(always)]
pub fn to_str_radix_special(num: f64, rdx: uint) -> (~str, bool) {
    strconv::to_str_common(&num, rdx, true,
                           strconv::SignNeg, strconv::DigAll)
}

/**
 * Converts a float to a string with exactly the number of
 * provided significant digits
 *
 * # Arguments
 *
 * * num - The float value
 * * digits - The number of significant digits
 */
#[inline(always)]
pub fn to_str_exact(num: f64, dig: uint) -> ~str {
    let (r, _) = strconv::to_str_common(
        &num, 10u, true, strconv::SignNeg, strconv::DigExact(dig));
    r
}

/**
 * Converts a float to a string with a maximum number of
 * significant digits
 *
 * # Arguments
 *
 * * num - The float value
 * * digits - The number of significant digits
 */
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

/**
 * Convert a string in base 10 to a float.
 * Accepts a optional decimal exponent.
 *
 * This function accepts strings such as
 *
 * * '3.14'
 * * '+3.14', equivalent to '3.14'
 * * '-3.14'
 * * '2.5E10', or equivalently, '2.5e10'
 * * '2.5E-10'
 * * '.' (understood as 0)
 * * '5.'
 * * '.5', or, equivalently,  '0.5'
 * * '+inf', 'inf', '-inf', 'NaN'
 *
 * Leading and trailing whitespace represent an error.
 *
 * # Arguments
 *
 * * num - A string
 *
 * # Return value
 *
 * `none` if the string did not represent a valid number.  Otherwise,
 * `Some(n)` where `n` is the floating-point number represented by `num`.
 */
#[inline(always)]
pub fn from_str(num: &str) -> Option<f64> {
    strconv::from_str_common(num, 10u, true, true, true,
                             strconv::ExpDec, false, false)
}

/**
 * Convert a string in base 16 to a float.
 * Accepts a optional binary exponent.
 *
 * This function accepts strings such as
 *
 * * 'a4.fe'
 * * '+a4.fe', equivalent to 'a4.fe'
 * * '-a4.fe'
 * * '2b.aP128', or equivalently, '2b.ap128'
 * * '2b.aP-128'
 * * '.' (understood as 0)
 * * 'c.'
 * * '.c', or, equivalently,  '0.c'
 * * '+inf', 'inf', '-inf', 'NaN'
 *
 * Leading and trailing whitespace represent an error.
 *
 * # Arguments
 *
 * * num - A string
 *
 * # Return value
 *
 * `none` if the string did not represent a valid number.  Otherwise,
 * `Some(n)` where `n` is the floating-point number represented by `[num]`.
 */
#[inline(always)]
pub fn from_str_hex(num: &str) -> Option<f64> {
    strconv::from_str_common(num, 16u, true, true, true,
                             strconv::ExpBin, false, false)
}

/**
 * Convert a string in an given base to a float.
 *
 * Due to possible conflicts, this function does **not** accept
 * the special values `inf`, `-inf`, `+inf` and `NaN`, **nor**
 * does it recognize exponents of any kind.
 *
 * Leading and trailing whitespace represent an error.
 *
 * # Arguments
 *
 * * num - A string
 * * radix - The base to use. Must lie in the range [2 .. 36]
 *
 * # Return value
 *
 * `none` if the string did not represent a valid number. Otherwise,
 * `Some(n)` where `n` is the floating-point number represented by `num`.
 */
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

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
