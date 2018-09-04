// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module provides constants which are specific to the implementation
//! of the `f32` floating point data type.
//!
//! *[See also the `f32` primitive type](../../std/primitive.f32.html).*
//!
//! Mathematically significant numbers are provided in the `consts` sub-module.

#![stable(feature = "rust1", since = "1.0.0")]

use cmp;
use mem;
use num::FpCategory;

/// The radix or base of the internal representation of `f32`.
#[stable(feature = "rust1", since = "1.0.0")]
pub const RADIX: u32 = 2;

/// Number of significant digits in base 2.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MANTISSA_DIGITS: u32 = 24;
/// Approximate number of significant digits in base 10.
#[stable(feature = "rust1", since = "1.0.0")]
pub const DIGITS: u32 = 6;

/// [Machine epsilon] value for `f32`.
///
/// This is the difference between `1.0` and the next largest representable number.
///
/// [Machine epsilon]: https://en.wikipedia.org/wiki/Machine_epsilon
#[stable(feature = "rust1", since = "1.0.0")]
pub const EPSILON: f32 = 1.19209290e-07_f32;

/// Smallest finite `f32` value.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MIN: f32 = -3.40282347e+38_f32;
/// Smallest positive normal `f32` value.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MIN_POSITIVE: f32 = 1.17549435e-38_f32;
/// Largest finite `f32` value.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MAX: f32 = 3.40282347e+38_f32;

/// One greater than the minimum possible normal power of 2 exponent.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MIN_EXP: i32 = -125;
/// Maximum possible power of 2 exponent.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MAX_EXP: i32 = 128;

/// Minimum possible normal power of 10 exponent.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MIN_10_EXP: i32 = -37;
/// Maximum possible power of 10 exponent.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MAX_10_EXP: i32 = 38;

/// Not a Number (NaN).
#[stable(feature = "rust1", since = "1.0.0")]
pub const NAN: f32 = 0.0_f32 / 0.0_f32;
/// Infinity (∞).
#[stable(feature = "rust1", since = "1.0.0")]
pub const INFINITY: f32 = 1.0_f32 / 0.0_f32;
/// Negative infinity (-∞).
#[stable(feature = "rust1", since = "1.0.0")]
pub const NEG_INFINITY: f32 = -1.0_f32 / 0.0_f32;

/// Basic mathematical constants.
#[stable(feature = "rust1", since = "1.0.0")]
pub mod consts {
    // FIXME: replace with mathematical constants from cmath.

    /// Archimedes' constant (π)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const PI: f32 = 3.14159265358979323846264338327950288_f32;

    /// π/2
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_2: f32 = 1.57079632679489661923132169163975144_f32;

    /// π/3
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_3: f32 = 1.04719755119659774615421446109316763_f32;

    /// π/4
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_4: f32 = 0.785398163397448309615660845819875721_f32;

    /// π/6
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_6: f32 = 0.52359877559829887307710723054658381_f32;

    /// π/8
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_8: f32 = 0.39269908169872415480783042290993786_f32;

    /// 1/π
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_1_PI: f32 = 0.318309886183790671537767526745028724_f32;

    /// 2/π
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_2_PI: f32 = 0.636619772367581343075535053490057448_f32;

    /// 2/sqrt(π)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_2_SQRT_PI: f32 = 1.12837916709551257389615890312154517_f32;

    /// sqrt(2)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const SQRT_2: f32 = 1.41421356237309504880168872420969808_f32;

    /// 1/sqrt(2)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_1_SQRT_2: f32 = 0.707106781186547524400844362104849039_f32;

    /// Euler's number (e)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const E: f32 = 2.71828182845904523536028747135266250_f32;

    /// log<sub>2</sub>(e)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const LOG2_E: f32 = 1.44269504088896340735992468100189214_f32;

    /// log<sub>2</sub>(10)
    #[unstable(feature = "extra_log_consts", issue = "50540")]
    pub const LOG2_10: f32 = 3.32192809488736234787031942948939018_f32;

    /// log<sub>10</sub>(e)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const LOG10_E: f32 = 0.434294481903251827651128918916605082_f32;

    /// log<sub>10</sub>(2)
    #[unstable(feature = "extra_log_consts", issue = "50540")]
    pub const LOG10_2: f32 = 0.301029995663981195213738894724493027_f32;

    /// ln(2)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const LN_2: f32 = 0.693147180559945309417232121458176568_f32;

    /// ln(10)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const LN_10: f32 = 2.30258509299404568401799145468436421_f32;
}

#[lang = "f32"]
#[cfg(not(test))]
impl f32 {
    /// Returns `true` if this value is `NaN` and false otherwise.
    ///
    /// ```
    /// use std::f32;
    ///
    /// let nan = f32::NAN;
    /// let f = 7.0_f32;
    ///
    /// assert!(nan.is_nan());
    /// assert!(!f.is_nan());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_nan(self) -> bool {
        self != self
    }

    /// Returns `true` if this value is positive infinity or negative infinity and
    /// false otherwise.
    ///
    /// ```
    /// use std::f32;
    ///
    /// let f = 7.0f32;
    /// let inf = f32::INFINITY;
    /// let neg_inf = f32::NEG_INFINITY;
    /// let nan = f32::NAN;
    ///
    /// assert!(!f.is_infinite());
    /// assert!(!nan.is_infinite());
    ///
    /// assert!(inf.is_infinite());
    /// assert!(neg_inf.is_infinite());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_infinite(self) -> bool {
        self == INFINITY || self == NEG_INFINITY
    }

    /// Returns `true` if this number is neither infinite nor `NaN`.
    ///
    /// ```
    /// use std::f32;
    ///
    /// let f = 7.0f32;
    /// let inf = f32::INFINITY;
    /// let neg_inf = f32::NEG_INFINITY;
    /// let nan = f32::NAN;
    ///
    /// assert!(f.is_finite());
    ///
    /// assert!(!nan.is_finite());
    /// assert!(!inf.is_finite());
    /// assert!(!neg_inf.is_finite());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_finite(self) -> bool {
        !(self.is_nan() || self.is_infinite())
    }

    /// Returns `true` if the number is neither zero, infinite,
    /// [subnormal][subnormal], or `NaN`.
    ///
    /// ```
    /// use std::f32;
    ///
    /// let min = f32::MIN_POSITIVE; // 1.17549435e-38f32
    /// let max = f32::MAX;
    /// let lower_than_min = 1.0e-40_f32;
    /// let zero = 0.0_f32;
    ///
    /// assert!(min.is_normal());
    /// assert!(max.is_normal());
    ///
    /// assert!(!zero.is_normal());
    /// assert!(!f32::NAN.is_normal());
    /// assert!(!f32::INFINITY.is_normal());
    /// // Values between `0` and `min` are Subnormal.
    /// assert!(!lower_than_min.is_normal());
    /// ```
    /// [subnormal]: https://en.wikipedia.org/wiki/Denormal_number
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_normal(self) -> bool {
        self.classify() == FpCategory::Normal
    }

    /// Returns the floating point category of the number. If only one property
    /// is going to be tested, it is generally faster to use the specific
    /// predicate instead.
    ///
    /// ```
    /// use std::num::FpCategory;
    /// use std::f32;
    ///
    /// let num = 12.4_f32;
    /// let inf = f32::INFINITY;
    ///
    /// assert_eq!(num.classify(), FpCategory::Normal);
    /// assert_eq!(inf.classify(), FpCategory::Infinite);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn classify(self) -> FpCategory {
        const EXP_MASK: u32 = 0x7f800000;
        const MAN_MASK: u32 = 0x007fffff;

        let bits = self.to_bits();
        match (bits & MAN_MASK, bits & EXP_MASK) {
            (0, 0) => FpCategory::Zero,
            (_, 0) => FpCategory::Subnormal,
            (0, EXP_MASK) => FpCategory::Infinite,
            (_, EXP_MASK) => FpCategory::Nan,
            _ => FpCategory::Normal,
        }
    }

    /// Returns `true` if and only if `self` has a positive sign, including `+0.0`, `NaN`s with
    /// positive sign bit and positive infinity.
    ///
    /// ```
    /// let f = 7.0_f32;
    /// let g = -7.0_f32;
    ///
    /// assert!(f.is_sign_positive());
    /// assert!(!g.is_sign_positive());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_sign_positive(self) -> bool {
        !self.is_sign_negative()
    }

    /// Returns `true` if and only if `self` has a negative sign, including `-0.0`, `NaN`s with
    /// negative sign bit and negative infinity.
    ///
    /// ```
    /// let f = 7.0f32;
    /// let g = -7.0f32;
    ///
    /// assert!(!f.is_sign_negative());
    /// assert!(g.is_sign_negative());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_sign_negative(self) -> bool {
        // IEEE754 says: isSignMinus(x) is true if and only if x has negative sign. isSignMinus
        // applies to zeros and NaNs as well.
        self.to_bits() & 0x8000_0000 != 0
    }

    /// Takes the reciprocal (inverse) of a number, `1/x`.
    ///
    /// ```
    /// use std::f32;
    ///
    /// let x = 2.0_f32;
    /// let abs_difference = (x.recip() - (1.0/x)).abs();
    ///
    /// assert!(abs_difference <= f32::EPSILON);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn recip(self) -> f32 {
        1.0 / self
    }

    /// Converts radians to degrees.
    ///
    /// ```
    /// use std::f32::{self, consts};
    ///
    /// let angle = consts::PI;
    ///
    /// let abs_difference = (angle.to_degrees() - 180.0).abs();
    ///
    /// assert!(abs_difference <= f32::EPSILON);
    /// ```
    #[stable(feature = "f32_deg_rad_conversions", since="1.7.0")]
    #[inline]
    pub fn to_degrees(self) -> f32 {
        // Use a constant for better precision.
        const PIS_IN_180: f32 = 57.2957795130823208767981548141051703_f32;
        self * PIS_IN_180
    }

    /// Converts degrees to radians.
    ///
    /// ```
    /// use std::f32::{self, consts};
    ///
    /// let angle = 180.0f32;
    ///
    /// let abs_difference = (angle.to_radians() - consts::PI).abs();
    ///
    /// assert!(abs_difference <= f32::EPSILON);
    /// ```
    #[stable(feature = "f32_deg_rad_conversions", since="1.7.0")]
    #[inline]
    pub fn to_radians(self) -> f32 {
        let value: f32 = consts::PI;
        self * (value / 180.0f32)
    }

    /// Returns the maximum of the two numbers.
    ///
    /// ```
    /// let x = 1.0f32;
    /// let y = 2.0f32;
    ///
    /// assert_eq!(x.max(y), y);
    /// ```
    ///
    /// If one of the arguments is NaN, then the other argument is returned.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn max(self, other: f32) -> f32 {
        // IEEE754 says: maxNum(x, y) is the canonicalized number y if x < y, x if y < x, the
        // canonicalized number if one operand is a number and the other a quiet NaN. Otherwise it
        // is either x or y, canonicalized (this means results might differ among implementations).
        // When either x or y is a signalingNaN, then the result is according to 6.2.
        //
        // Since we do not support sNaN in Rust yet, we do not need to handle them.
        // FIXME(nagisa): due to https://bugs.llvm.org/show_bug.cgi?id=33303 we canonicalize by
        // multiplying by 1.0. Should switch to the `canonicalize` when it works.
        (if self.is_nan() || self < other { other } else { self }) * 1.0
    }

    /// Returns the minimum of the two numbers.
    ///
    /// ```
    /// let x = 1.0f32;
    /// let y = 2.0f32;
    ///
    /// assert_eq!(x.min(y), x);
    /// ```
    ///
    /// If one of the arguments is NaN, then the other argument is returned.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn min(self, other: f32) -> f32 {
        // IEEE754 says: minNum(x, y) is the canonicalized number x if x < y, y if y < x, the
        // canonicalized number if one operand is a number and the other a quiet NaN. Otherwise it
        // is either x or y, canonicalized (this means results might differ among implementations).
        // When either x or y is a signalingNaN, then the result is according to 6.2.
        //
        // Since we do not support sNaN in Rust yet, we do not need to handle them.
        // FIXME(nagisa): due to https://bugs.llvm.org/show_bug.cgi?id=33303 we canonicalize by
        // multiplying by 1.0. Should switch to the `canonicalize` when it works.
        (if other.is_nan() || self < other { self } else { other }) * 1.0
    }

    /// Raw transmutation to `u32`.
    ///
    /// This is currently identical to `transmute::<f32, u32>(self)` on all platforms.
    ///
    /// See `from_bits` for some discussion of the portability of this operation
    /// (there are almost no issues).
    ///
    /// Note that this function is distinct from `as` casting, which attempts to
    /// preserve the *numeric* value, and not the bitwise value.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_ne!((1f32).to_bits(), 1f32 as u32); // to_bits() is not casting!
    /// assert_eq!((12.5f32).to_bits(), 0x41480000);
    ///
    /// ```
    #[stable(feature = "float_bits_conv", since = "1.20.0")]
    #[inline]
    pub fn to_bits(self) -> u32 {
        unsafe { mem::transmute(self) }
    }

    /// Raw transmutation from `u32`.
    ///
    /// This is currently identical to `transmute::<u32, f32>(v)` on all platforms.
    /// It turns out this is incredibly portable, for two reasons:
    ///
    /// * Floats and Ints have the same endianness on all supported platforms.
    /// * IEEE-754 very precisely specifies the bit layout of floats.
    ///
    /// However there is one caveat: prior to the 2008 version of IEEE-754, how
    /// to interpret the NaN signaling bit wasn't actually specified. Most platforms
    /// (notably x86 and ARM) picked the interpretation that was ultimately
    /// standardized in 2008, but some didn't (notably MIPS). As a result, all
    /// signaling NaNs on MIPS are quiet NaNs on x86, and vice-versa.
    ///
    /// Rather than trying to preserve signaling-ness cross-platform, this
    /// implementation favours preserving the exact bits. This means that
    /// any payloads encoded in NaNs will be preserved even if the result of
    /// this method is sent over the network from an x86 machine to a MIPS one.
    ///
    /// If the results of this method are only manipulated by the same
    /// architecture that produced them, then there is no portability concern.
    ///
    /// If the input isn't NaN, then there is no portability concern.
    ///
    /// If you don't care about signalingness (very likely), then there is no
    /// portability concern.
    ///
    /// Note that this function is distinct from `as` casting, which attempts to
    /// preserve the *numeric* value, and not the bitwise value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f32;
    /// let v = f32::from_bits(0x41480000);
    /// let difference = (v - 12.5).abs();
    /// assert!(difference <= 1e-5);
    /// ```
    #[stable(feature = "float_bits_conv", since = "1.20.0")]
    #[inline]
    pub fn from_bits(v: u32) -> Self {
        // It turns out the safety issues with sNaN were overblown! Hooray!
        unsafe { mem::transmute(v) }
    }

    /// A 3-way version of the IEEE `totalOrder` predicate.
    ///
    /// This is most commonly used with `cmp::Total`, not directly.
    ///
    /// This is useful in situations where you run into the fact that `f32` is
    /// `PartialOrd`, but not `Ord`, like sorting.  If you need a total order
    /// on arbitrary floating-point numbers you should use this -- or one of
    /// the related `total_*` methods -- they're implemented efficiently using
    /// just a few bitwise operations.
    ///
    /// This function mostly agrees with `PartialOrd` and the usual comparison
    /// operators in how it orders floating point numbers, but it additionally
    /// distinguishes negative from positive zero (it considers -0 as less than
    /// +0, whereas `==` considers them equal) and it orders Not-a-Number values
    /// relative to the numbers and distinguishes NaNs with different bit patterns.
    ///
    /// NaNs with positive sign are ordered greater than all other floating-point
    /// values including positive infinity, while NaNs with negative sign are
    /// ordered the least, below negative infinity. Two different NaNs with the
    /// same sign are ordered first by whether the are signaling (signaling is
    /// less than quiet if the sign is positive, reverse if negative) and second
    /// by their payload interpreted as integer (reversed order if the sign is negative).
    ///
    /// This means all different (canonical) floating point bit patterns are
    /// placed in a linear order, given below in ascending order:
    ///
    /// - Quiet Not-a-Number with negative sign (ordered by payload, descending)
    /// - Signaling Not-a-Number with negative sign (ordered by payload, descending)
    /// - Negative infinity
    /// - Negative finite non-zero numbers (ordered in the usual way)
    /// - Negative zero
    /// - Positive zero
    /// - Positive finite non-zero numbers (ordered in the usual way)
    /// - Positive infinity
    /// - Signaling Not-a-Number with positive sign (ordered by payload, ascending)
    /// - Quiet Not-a-Number with positive sign (ordered by payload, ascending)
    ///
    /// # Examples
    ///
    /// Most follow the normal ordering:
    /// ```
    /// #![feature(float_total_cmp)]
    /// use std::cmp::Ordering::*;
    ///
    /// assert_eq!(f32::total_cmp(1.0, 2.0), Less);
    /// assert_eq!(f32::total_cmp(1.0, 1.0), Equal);
    /// assert_eq!(f32::total_cmp(2.0, 1.0), Greater);
    /// assert_eq!(f32::total_cmp(-1.0, 1.0), Less);
    /// assert_eq!(f32::total_cmp(1.0, -1.0), Greater);
    /// assert_eq!(f32::total_cmp(-1.0, -2.0), Greater);
    /// assert_eq!(f32::total_cmp(-1.0, -1.0), Equal);
    /// assert_eq!(f32::total_cmp(-2.0, -1.0), Less);
    /// assert_eq!(f32::total_cmp(-std::f32::MAX, -1.0e9), Less);
    /// assert_eq!(f32::total_cmp(std::f32::MAX, 1.0e9), Greater);
    /// ```
    ///
    /// Zeros and NaNs:
    /// ```
    /// #![feature(float_total_cmp)]
    /// use std::cmp::Ordering::*;
    ///
    /// assert_eq!(f32::partial_cmp(&0.0, &-0.0), Some(Equal));
    /// assert_eq!(f32::total_cmp(-0.0, 0.0), Less);
    /// assert_eq!(f32::total_cmp(0.0, -0.0), Greater);
    ///
    /// assert_eq!(f32::partial_cmp(&std::f32::NAN, &std::f32::NAN), None);
    /// assert_eq!(f32::total_cmp(-std::f32::NAN, std::f32::NAN), Less);
    /// assert_eq!(f32::total_cmp(std::f32::NAN, std::f32::NAN), Equal);
    /// assert_eq!(f32::total_cmp(std::f32::NAN, -std::f32::NAN), Greater);
    /// assert_eq!(f32::total_cmp(std::f32::NAN, std::f32::INFINITY), Greater);
    /// assert_eq!(f32::total_cmp(-std::f32::NAN, -std::f32::INFINITY), Less);
    /// ```
    #[unstable(feature = "float_total_cmp", issue = "55339")]
    #[inline]
    pub fn total_cmp(self, other: f32) -> cmp::Ordering {
        self.total_cmp_key().cmp(&other.total_cmp_key())
    }

    #[inline]
    fn total_cmp_key(self) -> i32 {
        let x = self.to_bits() as i32;
        // Flip the bottom 31 bits if the high bit is set
        x ^ ((x >> 31) as u32 >> 1) as i32
    }
}

#[unstable(feature = "float_total_cmp", issue = "55339")]
impl PartialEq for cmp::Total<f32> {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

#[unstable(feature = "float_total_cmp", issue = "55339")]
impl Eq for cmp::Total<f32> {}

#[unstable(feature = "float_total_cmp", issue = "55339")]
impl PartialOrd for cmp::Total<f32> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// See `f32::total_cmp` for details.
///
/// Using this to sort floats:
/// ```
/// #![feature(float_total_cmp)]
///
/// let mut a = [
///     1.0,
///     -1.0,
///     0.0,
///     -0.0,
///     std::f32::NAN,
///     -std::f32::NAN,
///     std::f32::INFINITY,
///     std::f32::NEG_INFINITY,
/// ];
/// a.sort_by_key(|x| std::cmp::Total(*x));
/// assert_eq!(
///     format!("{:?}", a),
///     "[NaN, -inf, -1.0, -0.0, 0.0, 1.0, inf, NaN]"
/// );
/// ```
#[unstable(feature = "float_total_cmp", issue = "55339")]
impl Ord for cmp::Total<f32> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.0.total_cmp(other.0)
    }
}
