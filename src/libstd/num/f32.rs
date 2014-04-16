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

#![allow(missing_doc)]

use prelude::*;

use default::Default;
use from_str::FromStr;
use libc::{c_int};
use num::{FPCategory, FPNaN, FPInfinite , FPZero, FPSubnormal, FPNormal};
use num::{Zero, One, Bounded, strconv};
use num;
use intrinsics;

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

// FIXME(#11621): These constants should be deprecated once CTFE is implemented
// in favour of calling their respective functions in `Bounded` and `Float`.

pub static RADIX: uint = 2u;

pub static MANTISSA_DIGITS: uint = 53u;
pub static DIGITS: uint = 15u;

pub static EPSILON: f64 = 2.220446e-16_f64;

// FIXME (#1433): this is wrong, replace with hexadecimal (%a) statics
// below.
pub static MIN_VALUE: f64 = 2.225074e-308_f64;
pub static MAX_VALUE: f64 = 1.797693e+308_f64;

pub static MIN_EXP: uint = -1021u;
pub static MAX_EXP: uint = 1024u;

pub static MIN_10_EXP: int = -307;
pub static MAX_10_EXP: int = 308;

pub static NAN: f32 = 0.0_f32/0.0_f32;
pub static INFINITY: f32 = 1.0_f32/0.0_f32;
pub static NEG_INFINITY: f32 = -1.0_f32/0.0_f32;

/// Various useful constants.
pub mod consts {
    // FIXME (requires Issue #1433 to fix): replace with mathematical
    // staticants from cmath.

    // FIXME(#11621): These constants should be deprecated once CTFE is
    // implemented in favour of calling their respective functions in `Float`.

    /// Archimedes' constant
    pub static PI: f32 = 3.14159265358979323846264338327950288_f32;

    /// pi/2.0
    pub static FRAC_PI_2: f32 = 1.57079632679489661923132169163975144_f32;

    /// pi/4.0
    pub static FRAC_PI_4: f32 = 0.785398163397448309615660845819875721_f32;

    /// 1.0/pi
    pub static FRAC_1_PI: f32 = 0.318309886183790671537767526745028724_f32;

    /// 2.0/pi
    pub static FRAC_2_PI: f32 = 0.636619772367581343075535053490057448_f32;

    /// 2.0/sqrt(pi)
    pub static FRAC_2_SQRTPI: f32 = 1.12837916709551257389615890312154517_f32;

    /// sqrt(2.0)
    pub static SQRT2: f32 = 1.41421356237309504880168872420969808_f32;

    /// 1.0/sqrt(2.0)
    pub static FRAC_1_SQRT2: f32 = 0.707106781186547524400844362104849039_f32;

    /// Euler's number
    pub static E: f32 = 2.71828182845904523536028747135266250_f32;

    /// log2(e)
    pub static LOG2_E: f32 = 1.44269504088896340735992468100189214_f32;

    /// log10(e)
    pub static LOG10_E: f32 = 0.434294481903251827651128918916605082_f32;

    /// ln(2.0)
    pub static LN_2: f32 = 0.693147180559945309417232121458176568_f32;

    /// ln(10.0)
    pub static LN_10: f32 = 2.30258509299404568401799145468436421_f32;
}

impl Num for f32 {}

#[cfg(not(test))]
impl Eq for f32 {
    #[inline]
    fn eq(&self, other: &f32) -> bool { (*self) == (*other) }
}

#[cfg(not(test))]
impl Ord for f32 {
    #[inline]
    fn lt(&self, other: &f32) -> bool { (*self) < (*other) }
    #[inline]
    fn le(&self, other: &f32) -> bool { (*self) <= (*other) }
    #[inline]
    fn ge(&self, other: &f32) -> bool { (*self) >= (*other) }
    #[inline]
    fn gt(&self, other: &f32) -> bool { (*self) > (*other) }
}

impl Default for f32 {
    #[inline]
    fn default() -> f32 { 0.0 }
}

impl Zero for f32 {
    #[inline]
    fn zero() -> f32 { 0.0 }

    /// Returns true if the number is equal to either `0.0` or `-0.0`
    #[inline]
    fn is_zero(&self) -> bool { *self == 0.0 || *self == -0.0 }
}

impl One for f32 {
    #[inline]
    fn one() -> f32 { 1.0 }
}

#[cfg(not(test))]
impl Add<f32,f32> for f32 {
    #[inline]
    fn add(&self, other: &f32) -> f32 { *self + *other }
}

#[cfg(not(test))]
impl Sub<f32,f32> for f32 {
    #[inline]
    fn sub(&self, other: &f32) -> f32 { *self - *other }
}

#[cfg(not(test))]
impl Mul<f32,f32> for f32 {
    #[inline]
    fn mul(&self, other: &f32) -> f32 { *self * *other }
}

#[cfg(not(test))]
impl Div<f32,f32> for f32 {
    #[inline]
    fn div(&self, other: &f32) -> f32 { *self / *other }
}

#[cfg(not(test))]
impl Rem<f32,f32> for f32 {
    #[inline]
    fn rem(&self, other: &f32) -> f32 { *self % *other }
}

#[cfg(not(test))]
impl Neg<f32> for f32 {
    #[inline]
    fn neg(&self) -> f32 { -*self }
}

impl Signed for f32 {
    /// Computes the absolute value. Returns `NAN` if the number is `NAN`.
    #[inline]
    fn abs(&self) -> f32 { unsafe{intrinsics::fabsf32(*self)} }

    /// The positive difference of two numbers. Returns `0.0` if the number is less than or
    /// equal to `other`, otherwise the difference between`self` and `other` is returned.
    #[inline]
    fn abs_sub(&self, other: &f32) -> f32 { unsafe{cmath::fdimf(*self, *other)} }

    /// # Returns
    ///
    /// - `1.0` if the number is positive, `+0.0` or `INFINITY`
    /// - `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
    /// - `NAN` if the number is NaN
    #[inline]
    fn signum(&self) -> f32 {
        if self.is_nan() { NAN } else { unsafe{intrinsics::copysignf32(1.0, *self)} }
    }

    /// Returns `true` if the number is positive, including `+0.0` and `INFINITY`
    #[inline]
    fn is_positive(&self) -> bool { *self > 0.0 || (1.0 / *self) == INFINITY }

    /// Returns `true` if the number is negative, including `-0.0` and `NEG_INFINITY`
    #[inline]
    fn is_negative(&self) -> bool { *self < 0.0 || (1.0 / *self) == NEG_INFINITY }
}

impl Round for f32 {
    /// Round half-way cases toward `NEG_INFINITY`
    #[inline]
    fn floor(&self) -> f32 { unsafe{intrinsics::floorf32(*self)} }

    /// Round half-way cases toward `INFINITY`
    #[inline]
    fn ceil(&self) -> f32 { unsafe{intrinsics::ceilf32(*self)} }

    /// Round half-way cases away from `0.0`
    #[inline]
    fn round(&self) -> f32 { unsafe{intrinsics::roundf32(*self)} }

    /// The integer part of the number (rounds towards `0.0`)
    #[inline]
    fn trunc(&self) -> f32 { unsafe{intrinsics::truncf32(*self)} }

    /// The fractional part of the number, satisfying:
    ///
    /// ```rust
    /// let x = 1.65f32;
    /// assert!(x == x.trunc() + x.fract())
    /// ```
    #[inline]
    fn fract(&self) -> f32 { *self - self.trunc() }
}

impl Bounded for f32 {
    #[inline]
    fn min_value() -> f32 { 1.17549435e-38 }

    #[inline]
    fn max_value() -> f32 { 3.40282347e+38 }
}

impl Primitive for f32 {}

impl Float for f32 {
    fn powi(&self, n: i32) -> f32 { unsafe{intrinsics::powif32(*self, n)} }

    #[inline]
    fn max(self, other: f32) -> f32 {
        unsafe { cmath::fmaxf(self, other) }
    }

    #[inline]
    fn min(self, other: f32) -> f32 {
        unsafe { cmath::fminf(self, other) }
    }

    #[inline]
    fn nan() -> f32 { 0.0 / 0.0 }

    #[inline]
    fn infinity() -> f32 { 1.0 / 0.0 }

    #[inline]
    fn neg_infinity() -> f32 { -1.0 / 0.0 }

    #[inline]
    fn neg_zero() -> f32 { -0.0 }

    /// Returns `true` if the number is NaN
    #[inline]
    fn is_nan(&self) -> bool { *self != *self }

    /// Returns `true` if the number is infinite
    #[inline]
    fn is_infinite(&self) -> bool {
        *self == Float::infinity() || *self == Float::neg_infinity()
    }

    /// Returns `true` if the number is neither infinite or NaN
    #[inline]
    fn is_finite(&self) -> bool {
        !(self.is_nan() || self.is_infinite())
    }

    /// Returns `true` if the number is neither zero, infinite, subnormal or NaN
    #[inline]
    fn is_normal(&self) -> bool {
        self.classify() == FPNormal
    }

    /// Returns the floating point category of the number. If only one property is going to
    /// be tested, it is generally faster to use the specific predicate instead.
    fn classify(&self) -> FPCategory {
        static EXP_MASK: u32 = 0x7f800000;
        static MAN_MASK: u32 = 0x007fffff;

        let bits: u32 = unsafe {::cast::transmute(*self)};
        match (bits & MAN_MASK, bits & EXP_MASK) {
            (0, 0)        => FPZero,
            (_, 0)        => FPSubnormal,
            (0, EXP_MASK) => FPInfinite,
            (_, EXP_MASK) => FPNaN,
            _             => FPNormal,
        }
    }

    #[inline]
    fn mantissa_digits(_: Option<f32>) -> uint { 24 }

    #[inline]
    fn digits(_: Option<f32>) -> uint { 6 }

    #[inline]
    fn epsilon() -> f32 { 1.19209290e-07 }

    #[inline]
    fn min_exp(_: Option<f32>) -> int { -125 }

    #[inline]
    fn max_exp(_: Option<f32>) -> int { 128 }

    #[inline]
    fn min_10_exp(_: Option<f32>) -> int { -37 }

    #[inline]
    fn max_10_exp(_: Option<f32>) -> int { 38 }

    /// Constructs a floating point number by multiplying `x` by 2 raised to the power of `exp`
    #[inline]
    fn ldexp(x: f32, exp: int) -> f32 { unsafe{cmath::ldexpf(x, exp as c_int)} }

    /// Breaks the number into a normalized fraction and a base-2 exponent, satisfying:
    ///
    /// - `self = x * pow(2, exp)`
    /// - `0.5 <= abs(x) < 1.0`
    #[inline]
    fn frexp(&self) -> (f32, int) {
        unsafe {
            let mut exp = 0;
            let x = cmath::frexpf(*self, &mut exp);
            (x, exp as int)
        }
    }

    /// Returns the exponential of the number, minus `1`, in a way that is accurate
    /// even if the number is close to zero
    #[inline]
    fn exp_m1(&self) -> f32 { unsafe{cmath::expm1f(*self)} }

    /// Returns the natural logarithm of the number plus `1` (`ln(1+n)`) more accurately
    /// than if the operations were performed separately
    #[inline]
    fn ln_1p(&self) -> f32 { unsafe{cmath::log1pf(*self)} }

    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding error. This
    /// produces a more accurate result with better performance than a separate multiplication
    /// operation followed by an add.
    #[inline]
    fn mul_add(&self, a: f32, b: f32) -> f32 { unsafe{intrinsics::fmaf32(*self, a, b)} }

    /// Returns the next representable floating-point value in the direction of `other`
    #[inline]
    fn next_after(&self, other: f32) -> f32 { unsafe{cmath::nextafterf(*self, other)} }

    /// Returns the mantissa, exponent and sign as integers.
    fn integer_decode(&self) -> (u64, i16, i8) {
        let bits: u32 = unsafe {
            ::cast::transmute(*self)
        };
        let sign: i8 = if bits >> 31 == 0 { 1 } else { -1 };
        let mut exponent: i16 = ((bits >> 23) & 0xff) as i16;
        let mantissa = if exponent == 0 {
            (bits & 0x7fffff) << 1
        } else {
            (bits & 0x7fffff) | 0x800000
        };
        // Exponent bias + mantissa shift
        exponent -= 127 + 23;
        (mantissa as u64, exponent, sign)
    }

    /// Archimedes' constant
    #[inline]
    fn pi() -> f32 { 3.14159265358979323846264338327950288 }

    /// 2.0 * pi
    #[inline]
    fn two_pi() -> f32 { 6.28318530717958647692528676655900576 }

    /// pi / 2.0
    #[inline]
    fn frac_pi_2() -> f32 { 1.57079632679489661923132169163975144 }

    /// pi / 3.0
    #[inline]
    fn frac_pi_3() -> f32 { 1.04719755119659774615421446109316763 }

    /// pi / 4.0
    #[inline]
    fn frac_pi_4() -> f32 { 0.785398163397448309615660845819875721 }

    /// pi / 6.0
    #[inline]
    fn frac_pi_6() -> f32 { 0.52359877559829887307710723054658381 }

    /// pi / 8.0
    #[inline]
    fn frac_pi_8() -> f32 { 0.39269908169872415480783042290993786 }

    /// 1 .0/ pi
    #[inline]
    fn frac_1_pi() -> f32 { 0.318309886183790671537767526745028724 }

    /// 2.0 / pi
    #[inline]
    fn frac_2_pi() -> f32 { 0.636619772367581343075535053490057448 }

    /// 2.0 / sqrt(pi)
    #[inline]
    fn frac_2_sqrtpi() -> f32 { 1.12837916709551257389615890312154517 }

    /// sqrt(2.0)
    #[inline]
    fn sqrt2() -> f32 { 1.41421356237309504880168872420969808 }

    /// 1.0 / sqrt(2.0)
    #[inline]
    fn frac_1_sqrt2() -> f32 { 0.707106781186547524400844362104849039 }

    /// Euler's number
    #[inline]
    fn e() -> f32 { 2.71828182845904523536028747135266250 }

    /// log2(e)
    #[inline]
    fn log2_e() -> f32 { 1.44269504088896340735992468100189214 }

    /// log10(e)
    #[inline]
    fn log10_e() -> f32 { 0.434294481903251827651128918916605082 }

    /// ln(2.0)
    #[inline]
    fn ln_2() -> f32 { 0.693147180559945309417232121458176568 }

    /// ln(10.0)
    #[inline]
    fn ln_10() -> f32 { 2.30258509299404568401799145468436421 }

    /// The reciprocal (multiplicative inverse) of the number
    #[inline]
    fn recip(&self) -> f32 { 1.0 / *self }

    #[inline]
    fn powf(&self, n: &f32) -> f32 { unsafe{intrinsics::powf32(*self, *n)} }

    #[inline]
    fn sqrt(&self) -> f32 { unsafe{intrinsics::sqrtf32(*self)} }

    #[inline]
    fn rsqrt(&self) -> f32 { self.sqrt().recip() }

    #[inline]
    fn cbrt(&self) -> f32 { unsafe{cmath::cbrtf(*self)} }

    #[inline]
    fn hypot(&self, other: &f32) -> f32 { unsafe{cmath::hypotf(*self, *other)} }

    #[inline]
    fn sin(&self) -> f32 { unsafe{intrinsics::sinf32(*self)} }

    #[inline]
    fn cos(&self) -> f32 { unsafe{intrinsics::cosf32(*self)} }

    #[inline]
    fn tan(&self) -> f32 { unsafe{cmath::tanf(*self)} }

    #[inline]
    fn asin(&self) -> f32 { unsafe{cmath::asinf(*self)} }

    #[inline]
    fn acos(&self) -> f32 { unsafe{cmath::acosf(*self)} }

    #[inline]
    fn atan(&self) -> f32 { unsafe{cmath::atanf(*self)} }

    #[inline]
    fn atan2(&self, other: &f32) -> f32 { unsafe{cmath::atan2f(*self, *other)} }

    /// Simultaneously computes the sine and cosine of the number
    #[inline]
    fn sin_cos(&self) -> (f32, f32) {
        (self.sin(), self.cos())
    }

    /// Returns the exponential of the number
    #[inline]
    fn exp(&self) -> f32 { unsafe{intrinsics::expf32(*self)} }

    /// Returns 2 raised to the power of the number
    #[inline]
    fn exp2(&self) -> f32 { unsafe{intrinsics::exp2f32(*self)} }

    /// Returns the natural logarithm of the number
    #[inline]
    fn ln(&self) -> f32 { unsafe{intrinsics::logf32(*self)} }

    /// Returns the logarithm of the number with respect to an arbitrary base
    #[inline]
    fn log(&self, base: &f32) -> f32 { self.ln() / base.ln() }

    /// Returns the base 2 logarithm of the number
    #[inline]
    fn log2(&self) -> f32 { unsafe{intrinsics::log2f32(*self)} }

    /// Returns the base 10 logarithm of the number
    #[inline]
    fn log10(&self) -> f32 { unsafe{intrinsics::log10f32(*self)} }

    #[inline]
    fn sinh(&self) -> f32 { unsafe{cmath::sinhf(*self)} }

    #[inline]
    fn cosh(&self) -> f32 { unsafe{cmath::coshf(*self)} }

    #[inline]
    fn tanh(&self) -> f32 { unsafe{cmath::tanhf(*self)} }

    /// Inverse hyperbolic sine
    ///
    /// # Returns
    ///
    /// - on success, the inverse hyperbolic sine of `self` will be returned
    /// - `self` if `self` is `0.0`, `-0.0`, `INFINITY`, or `NEG_INFINITY`
    /// - `NAN` if `self` is `NAN`
    #[inline]
    fn asinh(&self) -> f32 {
        match *self {
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
    fn acosh(&self) -> f32 {
        match *self {
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
    fn atanh(&self) -> f32 {
        0.5 * ((2.0 * *self) / (1.0 - *self)).ln_1p()
    }

    /// Converts to degrees, assuming the number is in radians
    #[inline]
    fn to_degrees(&self) -> f32 { *self * (180.0f32 / Float::pi()) }

    /// Converts to radians, assuming the number is in degrees
    #[inline]
    fn to_radians(&self) -> f32 {
        let value: f32 = Float::pi();
        *self * (value / 180.0f32)
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
pub fn to_str(num: f32) -> ~str {
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
pub fn to_str_hex(num: f32) -> ~str {
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
pub fn to_str_radix_special(num: f32, rdx: uint) -> (~str, bool) {
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
pub fn to_str_exact(num: f32, dig: uint) -> ~str {
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
pub fn to_str_digits(num: f32, dig: uint) -> ~str {
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
pub fn to_str_exp_exact(num: f32, dig: uint, upper: bool) -> ~str {
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
pub fn to_str_exp_digits(num: f32, dig: uint, upper: bool) -> ~str {
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
        assert_eq!(Float::ldexp(1f32, -123), f1);
        assert_eq!(Float::ldexp(1f32, -111), f2);

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
        let f1: f32 = from_str_hex("1p-123").unwrap();
        let f2: f32 = from_str_hex("1p-111").unwrap();
        let (x1, exp1) = f1.frexp();
        let (x2, exp2) = f2.frexp();
        assert_eq!((x1, exp1), (0.5f32, -122));
        assert_eq!((x2, exp2), (0.5f32, -110));
        assert_eq!(Float::ldexp(x1, exp1), f1);
        assert_eq!(Float::ldexp(x2, exp2), f2);

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
        assert_eq!(2f32.powf(&100.0).integer_decode(), (8388608u64, 77i16, 1i8));
        assert_eq!(0f32.integer_decode(), (0u64, -150i16, 1i8));
        assert_eq!((-0f32).integer_decode(), (0u64, -150i16, -1i8));
        assert_eq!(INFINITY.integer_decode(), (8388608u64, 105i16, 1i8));
        assert_eq!(NEG_INFINITY.integer_decode(), (8388608u64, 105i16, -1i8));
        assert_eq!(NAN.integer_decode(), (12582912u64, 105i16, 1i8));
    }
}
