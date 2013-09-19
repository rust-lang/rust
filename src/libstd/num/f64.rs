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

#[allow(missing_doc)];
#[allow(non_uppercase_statics)];

use default::Default;
use libc::c_int;
use num::{Zero, One, strconv};
use num::{FPCategory, FPNaN, FPInfinite , FPZero, FPSubnormal, FPNormal};
use num;
use prelude::*;
use to_str;

pub use cmath::c_double_targ_consts::*;
pub use cmp::{min, max};

use self::delegated::*;

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
        // An inner module is required to get the #[inline] attribute on the
        // functions.
        mod delegated {
            use cmath::c_double_utils;
            use libc::{c_double, c_int};
            use unstable::intrinsics;

            $(
                #[inline] #[fixed_stack_segment] #[inline(never)]
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
    fn exp_m1(n: c_double) -> c_double = c_double_utils::exp_m1,
    fn abs_sub(a: c_double, b: c_double) -> c_double = c_double_utils::abs_sub,
    fn next_after(x: c_double, y: c_double) -> c_double = c_double_utils::next_after,
    fn frexp(n: c_double, value: &mut c_int) -> c_double = c_double_utils::frexp,
    fn hypot(x: c_double, y: c_double) -> c_double = c_double_utils::hypot,
    fn ldexp(x: c_double, n: c_int) -> c_double = c_double_utils::ldexp,
    fn lgamma(n: c_double, sign: &mut c_int) -> c_double = c_double_utils::lgamma,
    fn log_radix(n: c_double) -> c_double = c_double_utils::log_radix,
    fn ln_1p(n: c_double) -> c_double = c_double_utils::ln_1p,
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

impl Num for f64 {}

#[cfg(not(test))]
impl Eq for f64 {
    #[inline]
    fn eq(&self, other: &f64) -> bool { (*self) == (*other) }
}

#[cfg(not(test))]
impl ApproxEq<f64> for f64 {
    #[inline]
    fn approx_epsilon() -> f64 { 1.0e-6 }

    #[inline]
    fn approx_eq(&self, other: &f64) -> bool {
        self.approx_eq_eps(other, &1.0e-6)
    }

    #[inline]
    fn approx_eq_eps(&self, other: &f64, approx_epsilon: &f64) -> bool {
        (*self - *other).abs() < *approx_epsilon
    }
}

#[cfg(not(test))]
impl Ord for f64 {
    #[inline]
    fn lt(&self, other: &f64) -> bool { (*self) < (*other) }
    #[inline]
    fn le(&self, other: &f64) -> bool { (*self) <= (*other) }
    #[inline]
    fn ge(&self, other: &f64) -> bool { (*self) >= (*other) }
    #[inline]
    fn gt(&self, other: &f64) -> bool { (*self) > (*other) }
}

impl Orderable for f64 {
    /// Returns `NaN` if either of the numbers are `NaN`.
    #[inline]
    fn min(&self, other: &f64) -> f64 {
        match () {
            _ if self.is_NaN()  => *self,
            _ if other.is_NaN() => *other,
            _ if *self < *other => *self,
            _                   => *other,
        }
    }

    /// Returns `NaN` if either of the numbers are `NaN`.
    #[inline]
    fn max(&self, other: &f64) -> f64 {
        match () {
            _ if self.is_NaN()  => *self,
            _ if other.is_NaN() => *other,
            _ if *self > *other => *self,
            _                   => *other,
        }
    }

    /// Returns the number constrained within the range `mn <= self <= mx`.
    /// If any of the numbers are `NaN` then `NaN` is returned.
    #[inline]
    fn clamp(&self, mn: &f64, mx: &f64) -> f64 {
        match () {
            _ if self.is_NaN()   => *self,
            _ if !(*self <= *mx) => *mx,
            _ if !(*self >= *mn) => *mn,
            _                    => *self,
        }
    }
}

impl Default for f64 {
    #[inline]
    fn default() -> f64 { 0.0 }
}

impl Zero for f64 {
    #[inline]
    fn zero() -> f64 { 0.0 }

    /// Returns true if the number is equal to either `0.0` or `-0.0`
    #[inline]
    fn is_zero(&self) -> bool { *self == 0.0 || *self == -0.0 }
}

impl One for f64 {
    #[inline]
    fn one() -> f64 { 1.0 }
}

#[cfg(not(test))]
impl Add<f64,f64> for f64 {
    #[inline]
    fn add(&self, other: &f64) -> f64 { *self + *other }
}
#[cfg(not(test))]
impl Sub<f64,f64> for f64 {
    #[inline]
    fn sub(&self, other: &f64) -> f64 { *self - *other }
}
#[cfg(not(test))]
impl Mul<f64,f64> for f64 {
    #[inline]
    fn mul(&self, other: &f64) -> f64 { *self * *other }
}
#[cfg(not(test))]
impl Div<f64,f64> for f64 {
    #[inline]
    fn div(&self, other: &f64) -> f64 { *self / *other }
}
#[cfg(not(test))]
impl Rem<f64,f64> for f64 {
    #[inline]
    fn rem(&self, other: &f64) -> f64 { *self % *other }
}
#[cfg(not(test))]
impl Neg<f64> for f64 {
    fn neg(&self) -> f64 { -*self }
}

impl Signed for f64 {
    /// Computes the absolute value. Returns `NaN` if the number is `NaN`.
    #[inline]
    fn abs(&self) -> f64 { abs(*self) }

    ///
    /// The positive difference of two numbers. Returns `0.0` if the number is less than or
    /// equal to `other`, otherwise the difference between`self` and `other` is returned.
    ///
    #[inline]
    fn abs_sub(&self, other: &f64) -> f64 { abs_sub(*self, *other) }

    ///
    /// # Returns
    ///
    /// - `1.0` if the number is positive, `+0.0` or `infinity`
    /// - `-1.0` if the number is negative, `-0.0` or `neg_infinity`
    /// - `NaN` if the number is NaN
    ///
    #[inline]
    fn signum(&self) -> f64 {
        if self.is_NaN() { NaN } else { copysign(1.0, *self) }
    }

    /// Returns `true` if the number is positive, including `+0.0` and `infinity`
    #[inline]
    fn is_positive(&self) -> bool { *self > 0.0 || (1.0 / *self) == infinity }

    /// Returns `true` if the number is negative, including `-0.0` and `neg_infinity`
    #[inline]
    fn is_negative(&self) -> bool { *self < 0.0 || (1.0 / *self) == neg_infinity }
}

impl Round for f64 {
    /// Round half-way cases toward `neg_infinity`
    #[inline]
    fn floor(&self) -> f64 { floor(*self) }

    /// Round half-way cases toward `infinity`
    #[inline]
    fn ceil(&self) -> f64 { ceil(*self) }

    /// Round half-way cases away from `0.0`
    #[inline]
    fn round(&self) -> f64 { round(*self) }

    /// The integer part of the number (rounds towards `0.0`)
    #[inline]
    fn trunc(&self) -> f64 { trunc(*self) }

    ///
    /// The fractional part of the number, satisfying:
    ///
    /// ~~~ {.rust}
    /// assert!(x == trunc(x) + fract(x))
    /// ~~~
    ///
    #[inline]
    fn fract(&self) -> f64 { *self - self.trunc() }
}

impl Fractional for f64 {
    /// The reciprocal (multiplicative inverse) of the number
    #[inline]
    fn recip(&self) -> f64 { 1.0 / *self }
}

impl Algebraic for f64 {
    #[inline]
    fn pow(&self, n: &f64) -> f64 { pow(*self, *n) }

    #[inline]
    fn sqrt(&self) -> f64 { sqrt(*self) }

    #[inline]
    fn rsqrt(&self) -> f64 { self.sqrt().recip() }

    #[inline]
    fn cbrt(&self) -> f64 { cbrt(*self) }

    #[inline]
    fn hypot(&self, other: &f64) -> f64 { hypot(*self, *other) }
}

impl Trigonometric for f64 {
    #[inline]
    fn sin(&self) -> f64 { sin(*self) }

    #[inline]
    fn cos(&self) -> f64 { cos(*self) }

    #[inline]
    fn tan(&self) -> f64 { tan(*self) }

    #[inline]
    fn asin(&self) -> f64 { asin(*self) }

    #[inline]
    fn acos(&self) -> f64 { acos(*self) }

    #[inline]
    fn atan(&self) -> f64 { atan(*self) }

    #[inline]
    fn atan2(&self, other: &f64) -> f64 { atan2(*self, *other) }

    /// Simultaneously computes the sine and cosine of the number
    #[inline]
    fn sin_cos(&self) -> (f64, f64) {
        (self.sin(), self.cos())
    }
}

impl Exponential for f64 {
    /// Returns the exponential of the number
    #[inline]
    fn exp(&self) -> f64 { exp(*self) }

    /// Returns 2 raised to the power of the number
    #[inline]
    fn exp2(&self) -> f64 { exp2(*self) }

    /// Returns the natural logarithm of the number
    #[inline]
    fn ln(&self) -> f64 { ln(*self) }

    /// Returns the logarithm of the number with respect to an arbitrary base
    #[inline]
    fn log(&self, base: &f64) -> f64 { self.ln() / base.ln() }

    /// Returns the base 2 logarithm of the number
    #[inline]
    fn log2(&self) -> f64 { log2(*self) }

    /// Returns the base 10 logarithm of the number
    #[inline]
    fn log10(&self) -> f64 { log10(*self) }
}

impl Hyperbolic for f64 {
    #[inline]
    fn sinh(&self) -> f64 { sinh(*self) }

    #[inline]
    fn cosh(&self) -> f64 { cosh(*self) }

    #[inline]
    fn tanh(&self) -> f64 { tanh(*self) }

    ///
    /// Inverse hyperbolic sine
    ///
    /// # Returns
    ///
    /// - on success, the inverse hyperbolic sine of `self` will be returned
    /// - `self` if `self` is `0.0`, `-0.0`, `infinity`, or `neg_infinity`
    /// - `NaN` if `self` is `NaN`
    ///
    #[inline]
    fn asinh(&self) -> f64 {
        match *self {
            neg_infinity => neg_infinity,
            x => (x + ((x * x) + 1.0).sqrt()).ln(),
        }
    }

    ///
    /// Inverse hyperbolic cosine
    ///
    /// # Returns
    ///
    /// - on success, the inverse hyperbolic cosine of `self` will be returned
    /// - `infinity` if `self` is `infinity`
    /// - `NaN` if `self` is `NaN` or `self < 1.0` (including `neg_infinity`)
    ///
    #[inline]
    fn acosh(&self) -> f64 {
        match *self {
            x if x < 1.0 => Float::NaN(),
            x => (x + ((x * x) - 1.0).sqrt()).ln(),
        }
    }

    ///
    /// Inverse hyperbolic tangent
    ///
    /// # Returns
    ///
    /// - on success, the inverse hyperbolic tangent of `self` will be returned
    /// - `self` if `self` is `0.0` or `-0.0`
    /// - `infinity` if `self` is `1.0`
    /// - `neg_infinity` if `self` is `-1.0`
    /// - `NaN` if the `self` is `NaN` or outside the domain of `-1.0 <= self <= 1.0`
    ///   (including `infinity` and `neg_infinity`)
    ///
    #[inline]
    fn atanh(&self) -> f64 {
        0.5 * ((2.0 * *self) / (1.0 - *self)).ln_1p()
    }
}

impl Real for f64 {
    /// Archimedes' constant
    #[inline]
    fn pi() -> f64 { 3.14159265358979323846264338327950288 }

    /// 2.0 * pi
    #[inline]
    fn two_pi() -> f64 { 6.28318530717958647692528676655900576 }

    /// pi / 2.0
    #[inline]
    fn frac_pi_2() -> f64 { 1.57079632679489661923132169163975144 }

    /// pi / 3.0
    #[inline]
    fn frac_pi_3() -> f64 { 1.04719755119659774615421446109316763 }

    /// pi / 4.0
    #[inline]
    fn frac_pi_4() -> f64 { 0.785398163397448309615660845819875721 }

    /// pi / 6.0
    #[inline]
    fn frac_pi_6() -> f64 { 0.52359877559829887307710723054658381 }

    /// pi / 8.0
    #[inline]
    fn frac_pi_8() -> f64 { 0.39269908169872415480783042290993786 }

    /// 1.0 / pi
    #[inline]
    fn frac_1_pi() -> f64 { 0.318309886183790671537767526745028724 }

    /// 2.0 / pi
    #[inline]
    fn frac_2_pi() -> f64 { 0.636619772367581343075535053490057448 }

    /// 2.0 / sqrt(pi)
    #[inline]
    fn frac_2_sqrtpi() -> f64 { 1.12837916709551257389615890312154517 }

    /// sqrt(2.0)
    #[inline]
    fn sqrt2() -> f64 { 1.41421356237309504880168872420969808 }

    /// 1.0 / sqrt(2.0)
    #[inline]
    fn frac_1_sqrt2() -> f64 { 0.707106781186547524400844362104849039 }

    /// Euler's number
    #[inline]
    fn e() -> f64 { 2.71828182845904523536028747135266250 }

    /// log2(e)
    #[inline]
    fn log2_e() -> f64 { 1.44269504088896340735992468100189214 }

    /// log10(e)
    #[inline]
    fn log10_e() -> f64 { 0.434294481903251827651128918916605082 }

    /// ln(2.0)
    #[inline]
    fn ln_2() -> f64 { 0.693147180559945309417232121458176568 }

    /// ln(10.0)
    #[inline]
    fn ln_10() -> f64 { 2.30258509299404568401799145468436421 }

    /// Converts to degrees, assuming the number is in radians
    #[inline]
    fn to_degrees(&self) -> f64 { *self * (180.0f64 / Real::pi()) }

    /// Converts to radians, assuming the number is in degrees
    #[inline]
    fn to_radians(&self) -> f64 {
        let value: f64 = Real::pi();
        *self * (value / 180.0)
    }
}

impl RealExt for f64 {
    #[inline]
    fn lgamma(&self) -> (int, f64) {
        let mut sign = 0;
        let result = lgamma(*self, &mut sign);
        (sign as int, result)
    }

    #[inline]
    fn tgamma(&self) -> f64 { tgamma(*self) }

    #[inline]
    fn j0(&self) -> f64 { j0(*self) }

    #[inline]
    fn j1(&self) -> f64 { j1(*self) }

    #[inline]
    fn jn(&self, n: int) -> f64 { jn(n as c_int, *self) }

    #[inline]
    fn y0(&self) -> f64 { y0(*self) }

    #[inline]
    fn y1(&self) -> f64 { y1(*self) }

    #[inline]
    fn yn(&self, n: int) -> f64 { yn(n as c_int, *self) }
}

impl Bounded for f64 {
    #[inline]
    fn min_value() -> f64 { 2.2250738585072014e-308 }

    #[inline]
    fn max_value() -> f64 { 1.7976931348623157e+308 }
}

impl Primitive for f64 {
    #[inline]
    fn bits(_: Option<f64>) -> uint { 64 }

    #[inline]
    fn bytes(_: Option<f64>) -> uint { Primitive::bits(Some(0f64)) / 8 }
}

impl Float for f64 {
    #[inline]
    fn NaN() -> f64 { 0.0 / 0.0 }

    #[inline]
    fn infinity() -> f64 { 1.0 / 0.0 }

    #[inline]
    fn neg_infinity() -> f64 { -1.0 / 0.0 }

    #[inline]
    fn neg_zero() -> f64 { -0.0 }

    /// Returns `true` if the number is NaN
    #[inline]
    fn is_NaN(&self) -> bool { *self != *self }

    /// Returns `true` if the number is infinite
    #[inline]
    fn is_infinite(&self) -> bool {
        *self == Float::infinity() || *self == Float::neg_infinity()
    }

    /// Returns `true` if the number is neither infinite or NaN
    #[inline]
    fn is_finite(&self) -> bool {
        !(self.is_NaN() || self.is_infinite())
    }

    /// Returns `true` if the number is neither zero, infinite, subnormal or NaN
    #[inline]
    fn is_normal(&self) -> bool {
        self.classify() == FPNormal
    }

    /// Returns the floating point category of the number. If only one property is going to
    /// be tested, it is generally faster to use the specific predicate instead.
    fn classify(&self) -> FPCategory {
        static EXP_MASK: u64 = 0x7ff0000000000000;
        static MAN_MASK: u64 = 0x000fffffffffffff;

        match (
            unsafe { ::cast::transmute::<f64,u64>(*self) } & MAN_MASK,
            unsafe { ::cast::transmute::<f64,u64>(*self) } & EXP_MASK,
        ) {
            (0, 0)        => FPZero,
            (_, 0)        => FPSubnormal,
            (0, EXP_MASK) => FPInfinite,
            (_, EXP_MASK) => FPNaN,
            _             => FPNormal,
        }
    }

    #[inline]
    fn mantissa_digits(_: Option<f64>) -> uint { 53 }

    #[inline]
    fn digits(_: Option<f64>) -> uint { 15 }

    #[inline]
    fn epsilon() -> f64 { 2.2204460492503131e-16 }

    #[inline]
    fn min_exp(_: Option<f64>) -> int { -1021 }

    #[inline]
    fn max_exp(_: Option<f64>) -> int { 1024 }

    #[inline]
    fn min_10_exp(_: Option<f64>) -> int { -307 }

    #[inline]
    fn max_10_exp(_: Option<f64>) -> int { 308 }

    /// Constructs a floating point number by multiplying `x` by 2 raised to the power of `exp`
    #[inline]
    fn ldexp(x: f64, exp: int) -> f64 {
        ldexp(x, exp as c_int)
    }

    ///
    /// Breaks the number into a normalized fraction and a base-2 exponent, satisfying:
    ///
    /// - `self = x * pow(2, exp)`
    /// - `0.5 <= abs(x) < 1.0`
    ///
    #[inline]
    fn frexp(&self) -> (f64, int) {
        let mut exp = 0;
        let x = frexp(*self, &mut exp);
        (x, exp as int)
    }

    ///
    /// Returns the exponential of the number, minus `1`, in a way that is accurate
    /// even if the number is close to zero
    ///
    #[inline]
    fn exp_m1(&self) -> f64 { exp_m1(*self) }

    ///
    /// Returns the natural logarithm of the number plus `1` (`ln(1+n)`) more accurately
    /// than if the operations were performed separately
    ///
    #[inline]
    fn ln_1p(&self) -> f64 { ln_1p(*self) }

    ///
    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding error. This
    /// produces a more accurate result with better performance than a separate multiplication
    /// operation followed by an add.
    ///
    #[inline]
    fn mul_add(&self, a: f64, b: f64) -> f64 {
        mul_add(*self, a, b)
    }

    /// Returns the next representable floating-point value in the direction of `other`
    #[inline]
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
#[inline]
pub fn to_str(num: f64) -> ~str {
    let (r, _) = strconv::float_to_str_common(
        num, 10u, true, strconv::SignNeg, strconv::DigAll);
    r
}

///
/// Converts a float to a string in hexadecimal format
///
/// # Arguments
///
/// * num - The float value
///
#[inline]
pub fn to_str_hex(num: f64) -> ~str {
    let (r, _) = strconv::float_to_str_common(
        num, 16u, true, strconv::SignNeg, strconv::DigAll);
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
#[inline]
pub fn to_str_radix_special(num: f64, rdx: uint) -> (~str, bool) {
    strconv::float_to_str_common(num, rdx, true,
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
#[inline]
pub fn to_str_exact(num: f64, dig: uint) -> ~str {
    let (r, _) = strconv::float_to_str_common(
        num, 10u, true, strconv::SignNeg, strconv::DigExact(dig));
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
#[inline]
pub fn to_str_digits(num: f64, dig: uint) -> ~str {
    let (r, _) = strconv::float_to_str_common(
        num, 10u, true, strconv::SignNeg, strconv::DigMax(dig));
    r
}

impl to_str::ToStr for f64 {
    #[inline]
    fn to_str(&self) -> ~str { to_str_digits(*self, 8) }
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
    /// Fails if called on a special value like `inf`, `-inf` or `NaN` due to
    /// possible misinterpretation of the result at higher bases. If those values
    /// are expected, use `to_str_radix_special()` instead.
    #[inline]
    fn to_str_radix(&self, rdx: uint) -> ~str {
        let (r, special) = strconv::float_to_str_common(
            *self, rdx, true, strconv::SignNeg, strconv::DigAll);
        if special { fail!("number has a special value, \
                          try to_str_radix_special() if those are expected") }
        r
    }
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
#[inline]
pub fn from_str_hex(num: &str) -> Option<f64> {
    strconv::from_str_common(num, 16u, true, true, true,
                             strconv::ExpBin, false, false)
}

impl FromStr for f64 {
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
    #[inline]
    fn from_str(val: &str) -> Option<f64> {
        strconv::from_str_common(val, 10u, true, true, true,
                                 strconv::ExpDec, false, false)
    }
}

impl num::FromStrRadix for f64 {
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
    #[inline]
    fn from_str_radix(val: &str, rdx: uint) -> Option<f64> {
        strconv::from_str_common(val, rdx, true, true, false,
                                 strconv::ExpNone, false, false)
    }
}

#[cfg(test)]
mod tests {
    use f64::*;
    use prelude::*;

    use num::*;
    use num;
    use sys;

    #[test]
    fn test_num() {
        num::test_num(10f64, 2f64);
    }

    #[test]
    fn test_min() {
        assert_eq!(1f64.min(&2f64), 1f64);
        assert_eq!(2f64.min(&1f64), 1f64);

        let nan: f64 = Float::NaN();
        assert!(1f64.min(&nan).is_NaN());
        assert!(nan.min(&1f64).is_NaN());
    }

    #[test]
    fn test_max() {
        assert_eq!(1f64.max(&2f64), 2f64);
        assert_eq!(2f64.max(&1f64), 2f64);

        let nan: f64 = Float::NaN();
        assert!(1f64.max(&nan).is_NaN());
        assert!(nan.max(&1f64).is_NaN());
    }

    #[test]
    fn test_clamp() {
        assert_eq!(1f64.clamp(&2f64, &4f64), 2f64);
        assert_eq!(8f64.clamp(&2f64, &4f64), 4f64);
        assert_eq!(3f64.clamp(&2f64, &4f64), 3f64);

        let nan: f64 = Float::NaN();
        assert!(3f64.clamp(&nan, &4f64).is_NaN());
        assert!(3f64.clamp(&2f64, &nan).is_NaN());
        assert!(nan.clamp(&2f64, &4f64).is_NaN());
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
        let nan: f64 = Float::NaN();
        assert_eq!(inf.asinh(), inf);
        assert_eq!(neg_inf.asinh(), neg_inf);
        assert!(nan.asinh().is_NaN());
        assert_approx_eq!(2.0f64.asinh(), 1.443635475178810342493276740273105f64);
        assert_approx_eq!((-2.0f64).asinh(), -1.443635475178810342493276740273105f64);
    }

    #[test]
    fn test_acosh() {
        assert_eq!(1.0f64.acosh(), 0.0f64);
        assert!(0.999f64.acosh().is_NaN());

        let inf: f64 = Float::infinity();
        let neg_inf: f64 = Float::neg_infinity();
        let nan: f64 = Float::NaN();
        assert_eq!(inf.acosh(), inf);
        assert!(neg_inf.acosh().is_NaN());
        assert!(nan.acosh().is_NaN());
        assert_approx_eq!(2.0f64.acosh(), 1.31695789692481670862504634730796844f64);
        assert_approx_eq!(3.0f64.acosh(), 1.76274717403908605046521864995958461f64);
    }

    #[test]
    fn test_atanh() {
        assert_eq!(0.0f64.atanh(), 0.0f64);
        assert_eq!((-0.0f64).atanh(), -0.0f64);

        let inf: f64 = Float::infinity();
        let neg_inf: f64 = Float::neg_infinity();
        let nan: f64 = Float::NaN();
        assert_eq!(1.0f64.atanh(), inf);
        assert_eq!((-1.0f64).atanh(), neg_inf);
        assert!(2f64.atanh().atanh().is_NaN());
        assert!((-2f64).atanh().atanh().is_NaN());
        assert!(inf.atanh().is_NaN());
        assert!(neg_inf.atanh().is_NaN());
        assert!(nan.atanh().is_NaN());
        assert_approx_eq!(0.5f64.atanh(), 0.54930614433405484569762261846126285f64);
        assert_approx_eq!((-0.5f64).atanh(), -0.54930614433405484569762261846126285f64);
    }

    #[test]
    fn test_real_consts() {
        let pi: f64 = Real::pi();
        let two_pi: f64 = Real::two_pi();
        let frac_pi_2: f64 = Real::frac_pi_2();
        let frac_pi_3: f64 = Real::frac_pi_3();
        let frac_pi_4: f64 = Real::frac_pi_4();
        let frac_pi_6: f64 = Real::frac_pi_6();
        let frac_pi_8: f64 = Real::frac_pi_8();
        let frac_1_pi: f64 = Real::frac_1_pi();
        let frac_2_pi: f64 = Real::frac_2_pi();
        let frac_2_sqrtpi: f64 = Real::frac_2_sqrtpi();
        let sqrt2: f64 = Real::sqrt2();
        let frac_1_sqrt2: f64 = Real::frac_1_sqrt2();
        let e: f64 = Real::e();
        let log2_e: f64 = Real::log2_e();
        let log10_e: f64 = Real::log10_e();
        let ln_2: f64 = Real::ln_2();
        let ln_10: f64 = Real::ln_10();

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
        assert_eq!(infinity.abs(), infinity);
        assert_eq!(1f64.abs(), 1f64);
        assert_eq!(0f64.abs(), 0f64);
        assert_eq!((-0f64).abs(), 0f64);
        assert_eq!((-1f64).abs(), 1f64);
        assert_eq!(neg_infinity.abs(), infinity);
        assert_eq!((1f64/neg_infinity).abs(), 0f64);
        assert!(NaN.abs().is_NaN());
    }

    #[test]
    fn test_abs_sub() {
        assert_eq!((-1f64).abs_sub(&1f64), 0f64);
        assert_eq!(1f64.abs_sub(&1f64), 0f64);
        assert_eq!(1f64.abs_sub(&0f64), 1f64);
        assert_eq!(1f64.abs_sub(&-1f64), 2f64);
        assert_eq!(neg_infinity.abs_sub(&0f64), 0f64);
        assert_eq!(infinity.abs_sub(&1f64), infinity);
        assert_eq!(0f64.abs_sub(&neg_infinity), infinity);
        assert_eq!(0f64.abs_sub(&infinity), 0f64);
    }

    #[test] #[ignore(cfg(windows))] // FIXME #8663
    fn test_abs_sub_nowin() {
        assert!(NaN.abs_sub(&-1f64).is_NaN());
        assert!(1f64.abs_sub(&NaN).is_NaN());
    }

    #[test]
    fn test_signum() {
        assert_eq!(infinity.signum(), 1f64);
        assert_eq!(1f64.signum(), 1f64);
        assert_eq!(0f64.signum(), 1f64);
        assert_eq!((-0f64).signum(), -1f64);
        assert_eq!((-1f64).signum(), -1f64);
        assert_eq!(neg_infinity.signum(), -1f64);
        assert_eq!((1f64/neg_infinity).signum(), -1f64);
        assert!(NaN.signum().is_NaN());
    }

    #[test]
    fn test_is_positive() {
        assert!(infinity.is_positive());
        assert!(1f64.is_positive());
        assert!(0f64.is_positive());
        assert!(!(-0f64).is_positive());
        assert!(!(-1f64).is_positive());
        assert!(!neg_infinity.is_positive());
        assert!(!(1f64/neg_infinity).is_positive());
        assert!(!NaN.is_positive());
    }

    #[test]
    fn test_is_negative() {
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
    fn test_approx_eq() {
        assert!(1.0f64.approx_eq(&1f64));
        assert!(0.9999999f64.approx_eq(&1f64));
        assert!(1.000001f64.approx_eq_eps(&1f64, &1.0e-5));
        assert!(1.0000001f64.approx_eq_eps(&1f64, &1.0e-6));
        assert!(!1.0000001f64.approx_eq_eps(&1f64, &1.0e-7));
    }

    #[test]
    fn test_primitive() {
        let none: Option<f64> = None;
        assert_eq!(Primitive::bits(none), sys::size_of::<f64>() * 8);
        assert_eq!(Primitive::bytes(none), sys::size_of::<f64>());
    }

    #[test]
    fn test_is_normal() {
        let nan: f64 = Float::NaN();
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
        let nan: f64 = Float::NaN();
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
        let nan: f64 = Float::NaN();
        assert_eq!(Float::ldexp(inf, -123), inf);
        assert_eq!(Float::ldexp(neg_inf, -123), neg_inf);
        assert!(Float::ldexp(nan, -123).is_NaN());
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
        let nan: f64 = Float::NaN();
        assert_eq!(match inf.frexp() { (x, _) => x }, inf)
        assert_eq!(match neg_inf.frexp() { (x, _) => x }, neg_inf)
        assert!(match nan.frexp() { (x, _) => x.is_NaN() })
    }
}
