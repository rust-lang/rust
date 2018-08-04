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
//! of the `f64` floating point data type.
//!
//! *[See also the `f64` primitive type](../../std/primitive.f64.html).*
//!
//! Mathematically significant numbers are provided in the `consts` sub-module.

#![stable(feature = "rust1", since = "1.0.0")]

use mem;
use num::FpCategory;

/// The radix or base of the internal representation of `f64`.
#[stable(feature = "rust1", since = "1.0.0")]
pub const RADIX: u32 = 2;

/// Number of significant digits in base 2.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MANTISSA_DIGITS: u32 = 53;
/// Approximate number of significant digits in base 10.
#[stable(feature = "rust1", since = "1.0.0")]
pub const DIGITS: u32 = 15;

/// [Machine epsilon] value for `f64`.
///
/// This is the difference between `1.0` and the next largest representable number.
///
/// [Machine epsilon]: https://en.wikipedia.org/wiki/Machine_epsilon
#[stable(feature = "rust1", since = "1.0.0")]
pub const EPSILON: f64 = 2.220_446_049_250_313_1e-16_f64;

/// Smallest finite `f64` value.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MIN: f64 = -1.797_693_134_862_315_7e+308_f64;
/// Smallest positive normal `f64` value.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MIN_POSITIVE: f64 = 2.225_073_858_507_201_4e-308_f64;
/// Largest finite `f64` value.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MAX: f64 = 1.797_693_134_862_315_7e+308_f64;

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
    pub const PI: f64 = 3.141_592_653_589_793_238_462_643_383_279_502_88_f64;

    /// π/2
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_2: f64 = 1.570_796_326_794_896_619_231_321_691_639_751_44_f64;

    /// π/3
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_3: f64 = 1.047_197_551_196_597_746_154_214_461_093_167_63_f64;

    /// π/4
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_4: f64 = 0.785_398_163_397_448_309_615_660_845_819_875_721_f64;

    /// π/6
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_6: f64 = 0.523_598_775_598_298_873_077_107_230_546_583_81_f64;

    /// π/8
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_PI_8: f64 = 0.392_699_081_698_724_154_807_830_422_909_937_86_f64;

    /// 1/π
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_1_PI: f64 = 0.318_309_886_183_790_671_537_767_526_745_028_724_f64;

    /// 2/π
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_2_PI: f64 = 0.636_619_772_367_581_343_075_535_053_490_057_448_f64;

    /// 2/sqrt(π)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_2_SQRT_PI: f64 = 1.128_379_167_095_512_573_896_158_903_121_545_17_f64;

    /// sqrt(2)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const SQRT_2: f64 = 1.414_213_562_373_095_048_801_688_724_209_698_08_f64;

    /// 1/sqrt(2)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FRAC_1_SQRT_2: f64 = 0.707_106_781_186_547_524_400_844_362_104_849_039_f64;

    /// Euler's number (e)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const E: f64 = 2.718_281_828_459_045_235_360_287_471_352_662_50_f64;

    /// log<sub>2</sub>(10)
    #[unstable(feature = "extra_log_consts", issue = "50540")]
    pub const LOG2_10: f64 = 3.321_928_094_887_362_347_870_319_429_489_390_18_f64;

    /// log<sub>2</sub>(e)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const LOG2_E: f64 = 1.442_695_040_888_963_407_359_924_681_001_892_14_f64;

    /// log<sub>10</sub>(2)
    #[unstable(feature = "extra_log_consts", issue = "50540")]
    pub const LOG10_2: f64 = 0.301_029_995_663_981_195_213_738_894_724_493_027_f64;

    /// log<sub>10</sub>(e)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const LOG10_E: f64 = 0.434_294_481_903_251_827_651_128_918_916_605_082_f64;

    /// ln(2)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const LN_2: f64 = 0.693_147_180_559_945_309_417_232_121_458_176_568_f64;

    /// ln(10)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const LN_10: f64 = 2.302_585_092_994_045_684_017_991_454_684_364_21_f64;
}

#[lang = "f64"]
#[cfg(not(test))]
impl f64 {
    /// Returns `true` if this value is `NaN` and false otherwise.
    ///
    /// ```
    /// use std::f64;
    ///
    /// let nan = f64::NAN;
    /// let f = 7.0_f64;
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
    /// use std::f64;
    ///
    /// let f = 7.0f64;
    /// let inf = f64::INFINITY;
    /// let neg_inf = f64::NEG_INFINITY;
    /// let nan = f64::NAN;
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
    /// use std::f64;
    ///
    /// let f = 7.0f64;
    /// let inf: f64 = f64::INFINITY;
    /// let neg_inf: f64 = f64::NEG_INFINITY;
    /// let nan: f64 = f64::NAN;
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
    /// use std::f64;
    ///
    /// let min = f64::MIN_POSITIVE; // 2.2250738585072014e-308f64
    /// let max = f64::MAX;
    /// let lower_than_min = 1.0e-308_f64;
    /// let zero = 0.0f64;
    ///
    /// assert!(min.is_normal());
    /// assert!(max.is_normal());
    ///
    /// assert!(!zero.is_normal());
    /// assert!(!f64::NAN.is_normal());
    /// assert!(!f64::INFINITY.is_normal());
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
    /// use std::f64;
    ///
    /// let num = 12.4_f64;
    /// let inf = f64::INFINITY;
    ///
    /// assert_eq!(num.classify(), FpCategory::Normal);
    /// assert_eq!(inf.classify(), FpCategory::Infinite);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn classify(self) -> FpCategory {
        const EXP_MASK: u64 = 0x7ff0_0000_0000_0000;
        const MAN_MASK: u64 = 0x000f_ffff_ffff_ffff;

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
    /// let f = 7.0_f64;
    /// let g = -7.0_f64;
    ///
    /// assert!(f.is_sign_positive());
    /// assert!(!g.is_sign_positive());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_sign_positive(self) -> bool {
        !self.is_sign_negative()
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_deprecated(since = "1.0.0", reason = "renamed to is_sign_positive")]
    #[inline]
    #[doc(hidden)]
    pub fn is_positive(self) -> bool {
        self.is_sign_positive()
    }

    /// Returns `true` if and only if `self` has a negative sign, including `-0.0`, `NaN`s with
    /// negative sign bit and negative infinity.
    ///
    /// ```
    /// let f = 7.0_f64;
    /// let g = -7.0_f64;
    ///
    /// assert!(!f.is_sign_negative());
    /// assert!(g.is_sign_negative());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_sign_negative(self) -> bool {
        self.to_bits() & 0x8000_0000_0000_0000 != 0
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_deprecated(since = "1.0.0", reason = "renamed to is_sign_negative")]
    #[inline]
    #[doc(hidden)]
    pub fn is_negative(self) -> bool {
        self.is_sign_negative()
    }

    /// Takes the reciprocal (inverse) of a number, `1/x`.
    ///
    /// ```
    /// let x = 2.0_f64;
    /// let abs_difference = (x.recip() - (1.0/x)).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn recip(self) -> f64 {
        1.0 / self
    }

    /// Converts radians to degrees.
    ///
    /// ```
    /// use std::f64::consts;
    ///
    /// let angle = consts::PI;
    ///
    /// let abs_difference = (angle.to_degrees() - 180.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn to_degrees(self) -> f64 {
        // The division here is correctly rounded with respect to the true
        // value of 180/π. (This differs from f32, where a constant must be
        // used to ensure a correctly rounded result.)
        self * (180.0f64 / consts::PI)
    }

    /// Converts degrees to radians.
    ///
    /// ```
    /// use std::f64::consts;
    ///
    /// let angle = 180.0_f64;
    ///
    /// let abs_difference = (angle.to_radians() - consts::PI).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn to_radians(self) -> f64 {
        let value: f64 = consts::PI;
        self * (value / 180.0)
    }

    /// Returns the maximum of the two numbers.
    ///
    /// ```
    /// let x = 1.0_f64;
    /// let y = 2.0_f64;
    ///
    /// assert_eq!(x.max(y), y);
    /// ```
    ///
    /// If one of the arguments is NaN, then the other argument is returned.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn max(self, other: f64) -> f64 {
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
    /// let x = 1.0_f64;
    /// let y = 2.0_f64;
    ///
    /// assert_eq!(x.min(y), x);
    /// ```
    ///
    /// If one of the arguments is NaN, then the other argument is returned.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn min(self, other: f64) -> f64 {
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

    /// Raw transmutation to `u64`.
    ///
    /// This is currently identical to `transmute::<f64, u64>(self)` on all platforms.
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
    /// assert!((1f64).to_bits() != 1f64 as u64); // to_bits() is not casting!
    /// assert_eq!((12.5f64).to_bits(), 0x4029000000000000);
    ///
    /// ```
    #[stable(feature = "float_bits_conv", since = "1.20.0")]
    #[inline]
    pub fn to_bits(self) -> u64 {
        unsafe { mem::transmute(self) }
    }

    /// Raw transmutation from `u64`.
    ///
    /// This is currently identical to `transmute::<u64, f64>(v)` on all platforms.
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
    /// use std::f64;
    /// let v = f64::from_bits(0x4029000000000000);
    /// let difference = (v - 12.5).abs();
    /// assert!(difference <= 1e-5);
    /// ```
    #[stable(feature = "float_bits_conv", since = "1.20.0")]
    #[inline]
    pub fn from_bits(v: u64) -> Self {
        // It turns out the safety issues with sNaN were overblown! Hooray!
        unsafe { mem::transmute(v) }
    }
}
