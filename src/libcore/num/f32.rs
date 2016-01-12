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

// FIXME: MIN_VALUE and MAX_VALUE literals are parsed as -inf and inf #14353
#![allow(overflowing_literals)]

#![stable(feature = "rust1", since = "1.0.0")]

use intrinsics;
use mem;
use num::Float;
use num::FpCategory as Fp;

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(missing_docs)]
pub const RADIX: u32 = 2;

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(missing_docs)]
pub const MANTISSA_DIGITS: u32 = 24;
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(missing_docs)]
pub const DIGITS: u32 = 6;

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(missing_docs)]
pub const EPSILON: f32 = 1.19209290e-07_f32;

/// Smallest finite f32 value
#[stable(feature = "rust1", since = "1.0.0")]
pub const MIN: f32 = -3.40282347e+38_f32;
/// Smallest positive, normalized f32 value
#[stable(feature = "rust1", since = "1.0.0")]
pub const MIN_POSITIVE: f32 = 1.17549435e-38_f32;
/// Largest finite f32 value
#[stable(feature = "rust1", since = "1.0.0")]
pub const MAX: f32 = 3.40282347e+38_f32;

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(missing_docs)]
pub const MIN_EXP: i32 = -125;
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(missing_docs)]
pub const MAX_EXP: i32 = 128;

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(missing_docs)]
pub const MIN_10_EXP: i32 = -37;
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(missing_docs)]
pub const MAX_10_EXP: i32 = 38;

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(missing_docs)]
pub const NAN: f32 = 0.0_f32/0.0_f32;
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(missing_docs)]
pub const INFINITY: f32 = 1.0_f32/0.0_f32;
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(missing_docs)]
pub const NEG_INFINITY: f32 = -1.0_f32/0.0_f32;

/// Basic mathematical constants.
#[stable(feature = "rust1", since = "1.0.0")]
pub mod consts {
    // FIXME: replace with mathematical constants from cmath.

    /// Archimedes' constant
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const PI: f32 = 3.14159265358979323846264338327950288_f32;

    /// pi/2.0
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_2: f32 = 1.57079632679489661923132169163975144_f32;

    /// pi/3.0
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_3: f32 = 1.04719755119659774615421446109316763_f32;

    /// pi/4.0
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_4: f32 = 0.785398163397448309615660845819875721_f32;

    /// pi/6.0
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_6: f32 = 0.52359877559829887307710723054658381_f32;

    /// pi/8.0
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_8: f32 = 0.39269908169872415480783042290993786_f32;

    /// 1.0/pi
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_1_PI: f32 = 0.318309886183790671537767526745028724_f32;

    /// 2.0/pi
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_2_PI: f32 = 0.636619772367581343075535053490057448_f32;

    /// 2.0/sqrt(pi)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_2_SQRT_PI: f32 = 1.12837916709551257389615890312154517_f32;

    /// sqrt(2.0)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const SQRT_2: f32 = 1.41421356237309504880168872420969808_f32;

    /// 1.0/sqrt(2.0)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_1_SQRT_2: f32 = 0.707106781186547524400844362104849039_f32;

    /// Euler's number
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const E: f32 = 2.71828182845904523536028747135266250_f32;

    /// log2(e)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const LOG2_E: f32 = 1.44269504088896340735992468100189214_f32;

    /// log10(e)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const LOG10_E: f32 = 0.434294481903251827651128918916605082_f32;

    /// ln(2.0)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const LN_2: f32 = 0.693147180559945309417232121458176568_f32;

    /// ln(10.0)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const LN_10: f32 = 2.30258509299404568401799145468436421_f32;
}

#[unstable(feature = "core_float",
           reason = "stable interface is via `impl f{32,64}` in later crates",
           issue = "27702")]
impl Float for f32 {
    #[inline]
    fn nan() -> f32 { NAN }

    #[inline]
    fn infinity() -> f32 { INFINITY }

    #[inline]
    fn neg_infinity() -> f32 { NEG_INFINITY }

    #[inline]
    fn zero() -> f32 { 0.0 }

    #[inline]
    fn neg_zero() -> f32 { -0.0 }

    #[inline]
    fn one() -> f32 { 1.0 }

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
        self.classify() == Fp::Normal
    }

    /// Returns the floating point category of the number. If only one property
    /// is going to be tested, it is generally faster to use the specific
    /// predicate instead.
    fn classify(self) -> Fp {
        const EXP_MASK: u32 = 0x7f800000;
        const MAN_MASK: u32 = 0x007fffff;

        let bits: u32 = unsafe { mem::transmute(self) };
        match (bits & MAN_MASK, bits & EXP_MASK) {
            (0, 0)        => Fp::Zero,
            (_, 0)        => Fp::Subnormal,
            (0, EXP_MASK) => Fp::Infinite,
            (_, EXP_MASK) => Fp::Nan,
            _             => Fp::Normal,
        }
    }

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

    /// Computes the absolute value of `self`. Returns `Float::nan()` if the
    /// number is `Float::nan()`.
    #[inline]
    #[cfg(stage0)]
    fn abs(self) -> f32 {
        unsafe { intrinsics::fabsf32(self) }
    }
    #[inline]
    #[cfg(not(stage0))]
    fn abs(self) -> f32 {
        unsafe { intrinsics::fabs(self) }
    }

    /// Returns a number that represents the sign of `self`.
    ///
    /// - `1.0` if the number is positive, `+0.0` or `Float::infinity()`
    /// - `-1.0` if the number is negative, `-0.0` or `Float::neg_infinity()`
    /// - `Float::nan()` if the number is `Float::nan()`
    #[inline]
    #[cfg(stage0)]
    fn signum(self) -> f32 {
        if self.is_nan() {
            Float::nan()
        } else {
            unsafe { intrinsics::copysignf32(1.0, self) }
        }
    }
    #[inline]
    #[cfg(not(stage0))]
    fn signum(self) -> f32 {
        if self.is_nan() {
            Float::nan()
        } else {
            unsafe { intrinsics::copysign(1.0, self) }
        }
    }

    /// Returns `true` if `self` is positive, including `+0.0` and
    /// `Float::infinity()`.
    #[inline]
    fn is_sign_positive(self) -> bool {
        self > 0.0 || (1.0 / self) == Float::infinity()
    }

    /// Returns `true` if `self` is negative, including `-0.0` and
    /// `Float::neg_infinity()`.
    #[inline]
    fn is_sign_negative(self) -> bool {
        self < 0.0 || (1.0 / self) == Float::neg_infinity()
    }

    /// Returns the reciprocal (multiplicative inverse) of the number.
    #[inline]
    fn recip(self) -> f32 { 1.0 / self }

    #[inline]
    #[cfg(stage0)]
    fn powi(self, n: i32) -> f32 {
        unsafe { intrinsics::powif32(self, n) }
    }
    #[inline]
    #[cfg(not(stage0))]
    fn powi(self, n: i32) -> f32 {
        unsafe { intrinsics::powi(self, n) }
    }

    /// Converts to degrees, assuming the number is in radians.
    #[inline]
    fn to_degrees(self) -> f32 { self * (180.0f32 / consts::PI) }

    /// Converts to radians, assuming the number is in degrees.
    #[inline]
    fn to_radians(self) -> f32 {
        let value: f32 = consts::PI;
        self * (value / 180.0f32)
    }
}
