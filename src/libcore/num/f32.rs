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

#![doc(primitive = "f32")]
// FIXME: MIN_VALUE and MAX_VALUE literals are parsed as -inf and inf #14353
#![allow(type_overflow)]

use intrinsics;
use mem;
use num::{FPNormal, FPCategory, FPZero, FPSubnormal, FPInfinite, FPNaN};
use num::Float;
use option::Option;

pub static RADIX: uint = 2u;

pub static MANTISSA_DIGITS: uint = 24u;
pub static DIGITS: uint = 6u;

pub static EPSILON: f32 = 1.19209290e-07_f32;

/// Smallest finite f32 value
pub static MIN_VALUE: f32 = -3.40282347e+38_f32;
/// Smallest positive, normalized f32 value
pub static MIN_POS_VALUE: f32 = 1.17549435e-38_f32;
/// Largest finite f32 value
pub static MAX_VALUE: f32 = 3.40282347e+38_f32;

pub static MIN_EXP: int = -125;
pub static MAX_EXP: int = 128;

pub static MIN_10_EXP: int = -37;
pub static MAX_10_EXP: int = 38;

pub static NAN: f32 = 0.0_f32/0.0_f32;
pub static INFINITY: f32 = 1.0_f32/0.0_f32;
pub static NEG_INFINITY: f32 = -1.0_f32/0.0_f32;

/// Various useful constants.
pub mod consts {
    // FIXME: replace with mathematical constants from cmath.

    // FIXME(#5527): These constants should be deprecated once associated
    // constants are implemented in favour of referencing the respective members
    // of `Float`.

    /// Archimedes' constant
    pub static PI: f32 = 3.14159265358979323846264338327950288_f32;

    /// pi * 2.0
    pub static PI_2: f32 = 6.28318530717958647692528676655900576_f32;

    /// pi/2.0
    pub static FRAC_PI_2: f32 = 1.57079632679489661923132169163975144_f32;

    /// pi/3.0
    pub static FRAC_PI_3: f32 = 1.04719755119659774615421446109316763_f32;

    /// pi/4.0
    pub static FRAC_PI_4: f32 = 0.785398163397448309615660845819875721_f32;

    /// pi/6.0
    pub static FRAC_PI_6: f32 = 0.52359877559829887307710723054658381_f32;

    /// pi/8.0
    pub static FRAC_PI_8: f32 = 0.39269908169872415480783042290993786_f32;

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

impl Float for f32 {
    #[inline]
    fn nan() -> f32 { NAN }

    #[inline]
    fn infinity() -> f32 { INFINITY }

    #[inline]
    fn neg_infinity() -> f32 { NEG_INFINITY }

    #[inline]
    fn neg_zero() -> f32 { -0.0 }

    /// Returns `true` if the number is NaN.
    #[inline]
    fn is_nan(self) -> bool { self != self }

    /// Returns `true` if the number is infinite.
    #[inline]
    fn is_infinite(self) -> bool {
        self == Float::infinity() || self == Float::neg_infinity()
    }

    /// Returns `true` if the number is neither infinite or NaN.
    #[inline]
    fn is_finite(self) -> bool {
        !(self.is_nan() || self.is_infinite())
    }

    /// Returns `true` if the number is neither zero, infinite, subnormal or NaN.
    #[inline]
    fn is_normal(self) -> bool {
        self.classify() == FPNormal
    }

    /// Returns the floating point category of the number. If only one property
    /// is going to be tested, it is generally faster to use the specific
    /// predicate instead.
    fn classify(self) -> FPCategory {
        static EXP_MASK: u32 = 0x7f800000;
        static MAN_MASK: u32 = 0x007fffff;

        let bits: u32 = unsafe { mem::transmute(self) };
        match (bits & MAN_MASK, bits & EXP_MASK) {
            (0, 0)        => FPZero,
            (_, 0)        => FPSubnormal,
            (0, EXP_MASK) => FPInfinite,
            (_, EXP_MASK) => FPNaN,
            _             => FPNormal,
        }
    }

    #[inline]
    fn mantissa_digits(_: Option<f32>) -> uint { MANTISSA_DIGITS }

    #[inline]
    fn digits(_: Option<f32>) -> uint { DIGITS }

    #[inline]
    fn epsilon() -> f32 { EPSILON }

    #[inline]
    fn min_exp(_: Option<f32>) -> int { MIN_EXP }

    #[inline]
    fn max_exp(_: Option<f32>) -> int { MAX_EXP }

    #[inline]
    fn min_10_exp(_: Option<f32>) -> int { MIN_10_EXP }

    #[inline]
    fn max_10_exp(_: Option<f32>) -> int { MAX_10_EXP }

    #[inline]
    fn min_pos_value(_: Option<f32>) -> f32 { MIN_POS_VALUE }

    /// Returns the mantissa, exponent and sign as integers.
    fn integer_decode(self) -> (u64, i16, i8) {
        let bits: u32 = unsafe { mem::transmute(self) };
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

    /// Rounds towards minus infinity.
    #[inline]
    fn floor(self) -> f32 {
        unsafe { intrinsics::floorf32(self) }
    }

    /// Rounds towards plus infinity.
    #[inline]
    fn ceil(self) -> f32 {
        unsafe { intrinsics::ceilf32(self) }
    }

    /// Rounds to nearest integer. Rounds half-way cases away from zero.
    #[inline]
    fn round(self) -> f32 {
        unsafe { intrinsics::roundf32(self) }
    }

    /// Returns the integer part of the number (rounds towards zero).
    #[inline]
    fn trunc(self) -> f32 {
        unsafe { intrinsics::truncf32(self) }
    }

    /// The fractional part of the number, satisfying:
    ///
    /// ```rust
    /// let x = 1.65f32;
    /// assert!(x == x.trunc() + x.fract())
    /// ```
    #[inline]
    fn fract(self) -> f32 { self - self.trunc() }

    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding
    /// error. This produces a more accurate result with better performance than
    /// a separate multiplication operation followed by an add.
    #[inline]
    fn mul_add(self, a: f32, b: f32) -> f32 {
        unsafe { intrinsics::fmaf32(self, a, b) }
    }

    /// Returns the reciprocal (multiplicative inverse) of the number.
    #[inline]
    fn recip(self) -> f32 { 1.0 / self }

    fn powi(self, n: i32) -> f32 {
        unsafe { intrinsics::powif32(self, n) }
    }

    #[inline]
    fn powf(self, n: f32) -> f32 {
        unsafe { intrinsics::powf32(self, n) }
    }

    /// sqrt(2.0)
    #[inline]
    fn sqrt2() -> f32 { consts::SQRT2 }

    /// 1.0 / sqrt(2.0)
    #[inline]
    fn frac_1_sqrt2() -> f32 { consts::FRAC_1_SQRT2 }

    #[inline]
    fn sqrt(self) -> f32 {
        unsafe { intrinsics::sqrtf32(self) }
    }

    #[inline]
    fn rsqrt(self) -> f32 { self.sqrt().recip() }

    /// Archimedes' constant
    #[inline]
    fn pi() -> f32 { consts::PI }

    /// 2.0 * pi
    #[inline]
    fn two_pi() -> f32 { consts::PI_2 }

    /// pi / 2.0
    #[inline]
    fn frac_pi_2() -> f32 { consts::FRAC_PI_2 }

    /// pi / 3.0
    #[inline]
    fn frac_pi_3() -> f32 { consts::FRAC_PI_3 }

    /// pi / 4.0
    #[inline]
    fn frac_pi_4() -> f32 { consts::FRAC_PI_4 }

    /// pi / 6.0
    #[inline]
    fn frac_pi_6() -> f32 { consts::FRAC_PI_6 }

    /// pi / 8.0
    #[inline]
    fn frac_pi_8() -> f32 { consts::FRAC_PI_8 }

    /// 1.0 / pi
    #[inline]
    fn frac_1_pi() -> f32 { consts::FRAC_1_PI }

    /// 2.0 / pi
    #[inline]
    fn frac_2_pi() -> f32 { consts::FRAC_2_PI }

    /// 2.0 / sqrt(pi)
    #[inline]
    fn frac_2_sqrtpi() -> f32 { consts::FRAC_2_SQRTPI }

    /// Euler's number
    #[inline]
    fn e() -> f32 { consts::E }

    /// log2(e)
    #[inline]
    fn log2_e() -> f32 { consts::LOG2_E }

    /// log10(e)
    #[inline]
    fn log10_e() -> f32 { consts::LOG10_E }

    /// ln(2.0)
    #[inline]
    fn ln_2() -> f32 { consts::LN_2 }

    /// ln(10.0)
    #[inline]
    fn ln_10() -> f32 { consts::LN_10 }

    /// Returns the exponential of the number.
    #[inline]
    fn exp(self) -> f32 {
        unsafe { intrinsics::expf32(self) }
    }

    /// Returns 2 raised to the power of the number.
    #[inline]
    fn exp2(self) -> f32 {
        unsafe { intrinsics::exp2f32(self) }
    }

    /// Returns the natural logarithm of the number.
    #[inline]
    fn ln(self) -> f32 {
        unsafe { intrinsics::logf32(self) }
    }

    /// Returns the logarithm of the number with respect to an arbitrary base.
    #[inline]
    fn log(self, base: f32) -> f32 { self.ln() / base.ln() }

    /// Returns the base 2 logarithm of the number.
    #[inline]
    fn log2(self) -> f32 {
        unsafe { intrinsics::log2f32(self) }
    }

    /// Returns the base 10 logarithm of the number.
    #[inline]
    fn log10(self) -> f32 {
        unsafe { intrinsics::log10f32(self) }
    }

    /// Converts to degrees, assuming the number is in radians.
    #[inline]
    fn to_degrees(self) -> f32 { self * (180.0f32 / Float::pi()) }

    /// Converts to radians, assuming the number is in degrees.
    #[inline]
    fn to_radians(self) -> f32 {
        let value: f32 = Float::pi();
        self * (value / 180.0f32)
    }
}
