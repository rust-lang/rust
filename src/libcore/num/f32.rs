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

use num::strconv;
use num;
use option::Option;
use from_str;
use to_str;

#[cfg(notest)] use cmp::{Eq, Ord};
#[cfg(stage0,notest)]
use ops::{Add, Sub, Mul, Div, Modulo, Neg};
#[cfg(stage1,notest)]
#[cfg(stage2,notest)]
#[cfg(stage3,notest)]
use ops::{Add, Sub, Mul, Quot, Rem, Neg};

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
    fn nextafter(x: c_float, y: c_float) -> c_float = c_float_utils::nextafter,
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
    fn tgamma(n: c_float) -> c_float = c_float_utils::tgamma)


// These are not defined inside consts:: for consistency with
// the integer types

pub static NaN: f32 = 0.0_f32/0.0_f32;

pub static infinity: f32 = 1.0_f32/0.0_f32;

pub static neg_infinity: f32 = -1.0_f32/0.0_f32;

#[inline(always)]
pub fn is_NaN(f: f32) -> bool { f != f }

#[inline(always)]
pub fn add(x: f32, y: f32) -> f32 { return x + y; }

#[inline(always)]
pub fn sub(x: f32, y: f32) -> f32 { return x - y; }

#[inline(always)]
pub fn mul(x: f32, y: f32) -> f32 { return x * y; }

#[inline(always)]
pub fn quot(x: f32, y: f32) -> f32 { return x / y; }

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

/// Returns true if `x` is a positive number, including +0.0f320 and +Infinity
#[inline(always)]
pub fn is_positive(x: f32) -> bool {
    x > 0.0f32 || (1.0f32/x) == infinity
}

/// Returns true if `x` is a negative number, including -0.0f320 and -Infinity
#[inline(always)]
pub fn is_negative(x: f32) -> bool {
    x < 0.0f32 || (1.0f32/x) == neg_infinity
}

/**
 * Returns true if `x` is a negative number, including -0.0f320 and -Infinity
 *
 * This is the same as `f32::is_negative`.
 */
#[inline(always)]
pub fn is_nonpositive(x: f32) -> bool {
  return x < 0.0f32 || (1.0f32/x) == neg_infinity;
}

/**
 * Returns true if `x` is a positive number, including +0.0f320 and +Infinity
 *
 * This is the same as `f32::is_positive`.)
 */
#[inline(always)]
pub fn is_nonnegative(x: f32) -> bool {
  return x > 0.0f32 || (1.0f32/x) == infinity;
}

/// Returns true if `x` is a zero number (positive or negative zero)
#[inline(always)]
pub fn is_zero(x: f32) -> bool {
    return x == 0.0f32 || x == -0.0f32;
}

/// Returns true if `x`is an infinite number
#[inline(always)]
pub fn is_infinite(x: f32) -> bool {
    return x == infinity || x == neg_infinity;
}

/// Returns true if `x`is a finite number
#[inline(always)]
pub fn is_finite(x: f32) -> bool {
    return !(is_NaN(x) || is_infinite(x));
}

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
pub fn signbit(x: f32) -> int {
    if is_negative(x) { return 1; } else { return 0; }
}

#[inline(always)]
pub fn logarithm(n: f32, b: f32) -> f32 {
    return log2(n) / log2(b);
}

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

impl num::Zero for f32 {
    #[inline(always)]
    fn zero() -> f32 { 0.0 }
}

impl num::One for f32 {
    #[inline(always)]
    fn one() -> f32 { 1.0 }
}

#[cfg(notest)]
impl Add<f32,f32> for f32 {
    fn add(&self, other: &f32) -> f32 { *self + *other }
}
#[cfg(notest)]
impl Sub<f32,f32> for f32 {
    fn sub(&self, other: &f32) -> f32 { *self - *other }
}
#[cfg(notest)]
impl Mul<f32,f32> for f32 {
    fn mul(&self, other: &f32) -> f32 { *self * *other }
}
#[cfg(stage0,notest)]
impl Div<f32,f32> for f32 {
    fn div(&self, other: &f32) -> f32 { *self / *other }
}
#[cfg(stage1,notest)]
#[cfg(stage2,notest)]
#[cfg(stage3,notest)]
impl Quot<f32,f32> for f32 {
    #[inline(always)]
    fn quot(&self, other: &f32) -> f32 { *self / *other }
}
#[cfg(stage0,notest)]
impl Modulo<f32,f32> for f32 {
    fn modulo(&self, other: &f32) -> f32 { *self % *other }
}
#[cfg(stage1,notest)]
#[cfg(stage2,notest)]
#[cfg(stage3,notest)]
impl Rem<f32,f32> for f32 {
    #[inline(always)]
    fn rem(&self, other: &f32) -> f32 { *self % *other }
}
#[cfg(notest)]
impl Neg<f32> for f32 {
    fn neg(&self) -> f32 { -*self }
}

impl num::Round for f32 {
    #[inline(always)]
    fn round(&self, mode: num::RoundMode) -> f32 {
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
    fn floor(&self) -> f32 { floor(*self) }
    #[inline(always)]
    fn ceil(&self) -> f32 { ceil(*self) }
    #[inline(always)]
    fn fract(&self) -> f32 {
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
pub fn to_str(num: f32) -> ~str {
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
pub fn to_str_hex(num: f32) -> ~str {
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
pub fn to_str_radix(num: f32, rdx: uint) -> ~str {
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
pub fn to_str_radix_special(num: f32, rdx: uint) -> (~str, bool) {
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
pub fn to_str_exact(num: f32, dig: uint) -> ~str {
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
pub fn from_str(num: &str) -> Option<f32> {
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
pub fn from_str_hex(num: &str) -> Option<f32> {
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

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
