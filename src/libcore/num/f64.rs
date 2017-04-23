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

// FIXME: MIN_VALUE and MAX_VALUE literals are parsed as -inf and inf #14353
#![allow(overflowing_literals)]

#![stable(feature = "rust1", since = "1.0.0")]

use intrinsics;
use mem;
use num::FpCategory as Fp;
use num::Float;

/// The radix or base of the internal representation of `f64`.
#[stable(feature = "rust1", since = "1.0.0")]
pub const RADIX: u32 = 2;

/// Number of significant digits in base 2.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MANTISSA_DIGITS: u32 = 53;
/// Approximate number of significant digits in base 10.
#[stable(feature = "rust1", since = "1.0.0")]
pub const DIGITS: u32 = 15;

/// Difference between `1.0` and the next largest representable number.
#[stable(feature = "rust1", since = "1.0.0")]
pub const EPSILON: f64 = 2.2204460492503131e-16_f64;

/// Smallest finite `f64` value.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MIN: f64 = -1.7976931348623157e+308_f64;
/// Smallest positive normal `f64` value.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MIN_POSITIVE: f64 = 2.2250738585072014e-308_f64;
/// Largest finite `f64` value.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MAX: f64 = 1.7976931348623157e+308_f64;

/// One greater than the minimum possible normal power of 2 exponent.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MIN_EXP: i32 = -1021;
/// Maximum possible power of 2 exponent.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MAX_EXP: i32 = 1024;

/// Minimum possible normal power of 10 exponent.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MIN_10_EXP: i32 = -307;
/// Maximum possible power of 10 exponent.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MAX_10_EXP: i32 = 308;

/// Not a Number (NaN).
#[stable(feature = "rust1", since = "1.0.0")]
pub const NAN: f64 = 0.0_f64 / 0.0_f64;
/// Infinity (∞).
#[stable(feature = "rust1", since = "1.0.0")]
pub const INFINITY: f64 = 1.0_f64 / 0.0_f64;
/// Negative infinity (-∞).
#[stable(feature = "rust1", since = "1.0.0")]
pub const NEG_INFINITY: f64 = -1.0_f64 / 0.0_f64;

/// Basic mathematical constants.
#[stable(feature = "rust1", since = "1.0.0")]
pub mod consts {
    // FIXME: replace with mathematical constants from cmath.

    /// Archimedes' constant (π)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const PI: f64 = 3.14159265358979323846264338327950288_f64;

    /// π/2
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_2: f64 = 1.57079632679489661923132169163975144_f64;

    /// π/3
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_3: f64 = 1.04719755119659774615421446109316763_f64;

    /// π/4
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_4: f64 = 0.785398163397448309615660845819875721_f64;

    /// π/6
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_6: f64 = 0.52359877559829887307710723054658381_f64;

    /// π/8
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_8: f64 = 0.39269908169872415480783042290993786_f64;

    /// 1/π
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_1_PI: f64 = 0.318309886183790671537767526745028724_f64;

    /// 2/π
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_2_PI: f64 = 0.636619772367581343075535053490057448_f64;

    /// 2/sqrt(π)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_2_SQRT_PI: f64 = 1.12837916709551257389615890312154517_f64;

    /// sqrt(2)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const SQRT_2: f64 = 1.41421356237309504880168872420969808_f64;

    /// 1/sqrt(2)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_1_SQRT_2: f64 = 0.707106781186547524400844362104849039_f64;

    /// Euler's number (e)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const E: f64 = 2.71828182845904523536028747135266250_f64;

    /// log<sub>2</sub>(e)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const LOG2_E: f64 = 1.44269504088896340735992468100189214_f64;

    /// log<sub>10</sub>(e)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const LOG10_E: f64 = 0.434294481903251827651128918916605082_f64;

    /// ln(2)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const LN_2: f64 = 0.693147180559945309417232121458176568_f64;

    /// ln(10)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const LN_10: f64 = 2.30258509299404568401799145468436421_f64;
}

#[unstable(feature = "core_float",
           reason = "stable interface is via `impl f{32,64}` in later crates",
           issue = "32110")]
impl Float for f64 {
    /// Returns `true` if the number is NaN.
    #[inline]
    fn is_nan(self) -> bool {
        self != self
    }

    /// Returns `true` if the number is infinite.
    #[inline]
    fn is_infinite(self) -> bool {
        self == INFINITY || self == NEG_INFINITY
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
        const EXP_MASK: u64 = 0x7ff0000000000000;
        const MAN_MASK: u64 = 0x000fffffffffffff;

        let bits: u64 = unsafe { mem::transmute(self) };
        match (bits & MAN_MASK, bits & EXP_MASK) {
            (0, 0) => Fp::Zero,
            (_, 0) => Fp::Subnormal,
            (0, EXP_MASK) => Fp::Infinite,
            (_, EXP_MASK) => Fp::Nan,
            _ => Fp::Normal,
        }
    }

    /// Computes the absolute value of `self`. Returns `Float::nan()` if the
    /// number is `Float::nan()`.
    #[inline]
    fn abs(self) -> f64 {
        unsafe { intrinsics::fabsf64(self) }
    }

    /// Returns a number that represents the sign of `self`.
    ///
    /// - `1.0` if the number is positive, `+0.0` or `Float::infinity()`
    /// - `-1.0` if the number is negative, `-0.0` or `Float::neg_infinity()`
    /// - `Float::nan()` if the number is `Float::nan()`
    #[inline]
    fn signum(self) -> f64 {
        if self.is_nan() {
            NAN
        } else {
            unsafe { intrinsics::copysignf64(1.0, self) }
        }
    }

    /// Returns `true` if `self` is positive, including `+0.0` and
    /// `Float::infinity()`.
    #[inline]
    fn is_sign_positive(self) -> bool {
        self > 0.0 || (1.0 / self) == INFINITY
    }

    /// Returns `true` if `self` is negative, including `-0.0` and
    /// `Float::neg_infinity()`.
    #[inline]
    fn is_sign_negative(self) -> bool {
        self < 0.0 || (1.0 / self) == NEG_INFINITY
    }

    /// Returns the reciprocal (multiplicative inverse) of the number.
    #[inline]
    fn recip(self) -> f64 {
        1.0 / self
    }

    #[inline]
    fn powi(self, n: i32) -> f64 {
        unsafe { intrinsics::powif64(self, n) }
    }

    /// Converts to degrees, assuming the number is in radians.
    #[inline]
    fn to_degrees(self) -> f64 {
        self * (180.0f64 / consts::PI)
    }

    /// Converts to radians, assuming the number is in degrees.
    #[inline]
    fn to_radians(self) -> f64 {
        let value: f64 = consts::PI;
        self * (value / 180.0)
    }
}
