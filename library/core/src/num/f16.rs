//! Constants for the `f16` half-precision floating point type.
//!
//! *[See also the `f16` primitive type][f16].*
//!
//! Mathematically significant numbers are provided in the `consts` sub-module.
//!
//! For the constants defined directly in this module
//! (as distinct from those defined in the `consts` sub-module),
//! new code should instead use the associated constants
//! defined directly on the `f16` type.

#![unstable(feature = "f16", issue = "116909")]

use crate::convert::FloatToInt;
#[cfg(not(test))]
use crate::intrinsics;
use crate::mem;
use crate::num::FpCategory;

/// Basic mathematical constants.
#[unstable(feature = "f16", issue = "116909")]
pub mod consts {
    // FIXME: replace with mathematical constants from cmath.

    /// Archimedes' constant (π)
    #[unstable(feature = "f16", issue = "116909")]
    pub const PI: f16 = 3.14159265358979323846264338327950288_f16;

    /// The full circle constant (τ)
    ///
    /// Equal to 2π.
    #[unstable(feature = "f16", issue = "116909")]
    pub const TAU: f16 = 6.28318530717958647692528676655900577_f16;

    /// π/2
    #[unstable(feature = "f16", issue = "116909")]
    pub const FRAC_PI_2: f16 = 1.57079632679489661923132169163975144_f16;

    /// π/3
    #[unstable(feature = "f16", issue = "116909")]
    pub const FRAC_PI_3: f16 = 1.04719755119659774615421446109316763_f16;

    /// π/4
    #[unstable(feature = "f16", issue = "116909")]
    pub const FRAC_PI_4: f16 = 0.785398163397448309615660845819875721_f16;

    /// π/6
    #[unstable(feature = "f16", issue = "116909")]
    pub const FRAC_PI_6: f16 = 0.52359877559829887307710723054658381_f16;

    /// π/8
    #[unstable(feature = "f16", issue = "116909")]
    pub const FRAC_PI_8: f16 = 0.39269908169872415480783042290993786_f16;

    /// 1/π
    #[unstable(feature = "f16", issue = "116909")]
    pub const FRAC_1_PI: f16 = 0.318309886183790671537767526745028724_f16;

    /// 2/π
    #[unstable(feature = "f16", issue = "116909")]
    pub const FRAC_2_PI: f16 = 0.636619772367581343075535053490057448_f16;

    /// 2/sqrt(π)
    #[unstable(feature = "f16", issue = "116909")]
    pub const FRAC_2_SQRT_PI: f16 = 1.12837916709551257389615890312154517_f16;

    /// sqrt(2)
    #[unstable(feature = "f16", issue = "116909")]
    pub const SQRT_2: f16 = 1.41421356237309504880168872420969808_f16;

    /// 1/sqrt(2)
    #[unstable(feature = "f16", issue = "116909")]
    pub const FRAC_1_SQRT_2: f16 = 0.707106781186547524400844362104849039_f16;

    /// Euler's number (e)
    #[unstable(feature = "f16", issue = "116909")]
    pub const E: f16 = 2.71828182845904523536028747135266250_f16;

    /// log<sub>2</sub>(10)
    #[unstable(feature = "f16", issue = "116909")]
    pub const LOG2_10: f16 = 3.32192809488736234787031942948939018_f16;

    /// log<sub>2</sub>(e)
    #[unstable(feature = "f16", issue = "116909")]
    pub const LOG2_E: f16 = 1.44269504088896340735992468100189214_f16;

    /// log<sub>10</sub>(2)
    #[unstable(feature = "f16", issue = "116909")]
    pub const LOG10_2: f16 = 0.301029995663981195213738894724493027_f16;

    /// log<sub>10</sub>(e)
    #[unstable(feature = "f16", issue = "116909")]
    pub const LOG10_E: f16 = 0.434294481903251827651128918916605082_f16;

    /// ln(2)
    #[unstable(feature = "f16", issue = "116909")]
    pub const LN_2: f16 = 0.693147180559945309417232121458176568_f16;

    /// ln(10)
    #[unstable(feature = "f16", issue = "116909")]
    pub const LN_10: f16 = 2.30258509299404568401799145468436421_f16;
}

#[cfg(not(test))]
impl f16 {
    /// The radix or base of the internal representation of `f16`.
    #[unstable(feature = "f16", issue = "116909")]
    pub const RADIX: u32 = 2;

    /// Number of significant digits in base 2.
    #[unstable(feature = "f16", issue = "116909")]
    pub const MANTISSA_DIGITS: u32 = 11;

    /// Approximate number of significant digits in base 10.
    #[unstable(feature = "f16", issue = "116909")]
    pub const DIGITS: u32 = 3;

    /// [Machine epsilon] value for `f16`.
    ///
    /// This is the difference between `1.0` and the next larger representable number.
    ///
    /// [Machine epsilon]: https://en.wikipedia.org/wiki/Machine_epsilon
    #[unstable(feature = "f16", issue = "116909")]
    pub const EPSILON: f16 = 0.0009765625_f16;

    /// Smallest finite `f16` value.
    #[unstable(feature = "f16", issue = "116909")]
    pub const MIN: f16 = -65504.0_f16;

    /// Smallest positive normal `f16` value.
    #[unstable(feature = "f16", issue = "116909")]
    pub const MIN_POSITIVE: f16 = 6.1035156e-5_f16;

    /// Largest finite `f16` value.
    #[unstable(feature = "f16", issue = "116909")]
    pub const MAX: f16 = 65504.0;

    /// One greater than the minimum possible normal power of 2 exponent.
    #[unstable(feature = "f16", issue = "116909")]
    pub const MIN_EXP: i32 = -13;

    /// Maximum possible power of 2 exponent.
    #[unstable(feature = "f16", issue = "116909")]
    pub const MAX_EXP: i32 = 16;

    /// Minimum possible normal power of 10 exponent.
    #[unstable(feature = "f16", issue = "116909")]
    pub const MIN_10_EXP: i32 = -4;
    /// Maximum possible power of 10 exponent.
    #[unstable(feature = "f16", issue = "116909")]
    pub const MAX_10_EXP: i32 = 4;

    /// Not a Number (NaN).
    ///
    /// Note that IEEE 754 doesn't define just a single NaN value;
    /// a plethora of bit patterns are considered to be NaN.
    /// Furthermore, the standard makes a difference
    /// between a "signaling" and a "quiet" NaN,
    /// and allows inspecting its "payload" (the unspecified bits in the bit pattern).
    /// This constant isn't guaranteed to equal to any specific NaN bitpattern,
    /// and the stability of its representation over Rust versions
    /// and target platforms isn't guaranteed.
    #[rustc_diagnostic_item = "f16_nan"]
    #[unstable(feature = "f16", issue = "116909")]
    pub const NAN: f16 = 0.0_f16 / 0.0_f16;

    /// Infinity (∞).
    #[unstable(feature = "f16", issue = "116909")]
    pub const INFINITY: f16 = 1.0_f16 / 0.0_f16;

    /// Negative infinity (−∞).
    #[unstable(feature = "f16", issue = "116909")]
    pub const NEG_INFINITY: f16 = -1.0_f16 / 0.0_f16;

    pub(crate) const EXP_MASK: u16 = 0x7c00;
    pub(crate) const MAN_MASK: u16 = 0x03ff;
    pub(crate) const TINY_BITS: u16 = 0x1; // Smallest positive f16.
    pub(crate) const NEG_TINY_BITS: u16 = 0x8001; // Smallest (in magnitude) negative f16.
    pub(crate) const SIGN_MASK: u16 = 1 << 15;
    pub(crate) const CLEAR_SIGN_MASK: u16 = !Self::SIGN_MASK;

    /// Returns `true` if this value is NaN.
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// let nan = f16::NAN;
    /// let f = 7.0_f16;
    ///
    /// assert!(nan.is_nan());
    /// assert!(!f.is_nan());
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "f16", issue = "116909")]
    #[rustc_const_unstable(feature = "const_float_classify", issue = "72505")]
    pub const fn is_nan(self) -> bool {
        self != self
    }

    // FIXME(#50145): `abs` is publicly unavailable in core due to
    // concerns about portability, so this implementation is for
    // private use internally.
    #[inline]
    #[rustc_const_unstable(feature = "const_float_classify", issue = "72505")]
    pub(crate) const fn abs_private(self) -> f16 {
        // SAFETY: This transmutation is fine. Probably. For the reasons std is using it.
        unsafe {
            mem::transmute::<u16, f16>(mem::transmute::<f16, u16>(self) & Self::CLEAR_SIGN_MASK)
        }
    }

    /// Returns `true` if this value is positive infinity or negative infinity, and
    /// `false` otherwise.
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// let f = 7.0f16;
    /// let inf = f16::INFINITY;
    /// let neg_inf = f16::NEG_INFINITY;
    /// let nan = f16::NAN;
    ///
    /// assert!(!f.is_infinite());
    /// assert!(!nan.is_infinite());
    ///
    /// assert!(inf.is_infinite());
    /// assert!(neg_inf.is_infinite());
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "f16", issue = "116909")]
    #[rustc_const_unstable(feature = "const_float_classify", issue = "72505")]
    pub const fn is_infinite(self) -> bool {
        // Getting clever with transmutation can result in incorrect answers on some FPUs
        // FIXME: alter the Rust <-> Rust calling convention to prevent this problem.
        // See https://github.com/rust-lang/rust/issues/72327
        (self == f16::INFINITY) | (self == f16::NEG_INFINITY)
    }

    /// Returns `true` if this number is neither infinite nor NaN.
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// let f = 7.0f16;
    /// let inf: f16 = f16::INFINITY;
    /// let neg_inf: f16 = f16::NEG_INFINITY;
    /// let nan: f16 = f16::NAN;
    ///
    /// assert!(f.is_finite());
    ///
    /// assert!(!nan.is_finite());
    /// assert!(!inf.is_finite());
    /// assert!(!neg_inf.is_finite());
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "f16", issue = "116909")]
    #[rustc_const_unstable(feature = "const_float_classify", issue = "72505")]
    pub const fn is_finite(self) -> bool {
        // There's no need to handle NaN separately: if self is NaN,
        // the comparison is not true, exactly as desired.
        self.abs_private() < Self::INFINITY
    }

    /// Returns `true` if the number is [subnormal].
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// let min = f16::MIN_POSITIVE; // 6.1035156e-5_f16_f16
    /// let max = f16::MAX;
    /// let lower_than_min = 1.0e-5_f16;
    /// let zero = 0.0_f16;
    ///
    /// assert!(!min.is_subnormal());
    /// assert!(!max.is_subnormal());
    ///
    /// assert!(!zero.is_subnormal());
    /// assert!(!f16::NAN.is_subnormal());
    /// assert!(!f16::INFINITY.is_subnormal());
    /// // Values between `0` and `min` are Subnormal.
    /// assert!(lower_than_min.is_subnormal());
    /// ```
    /// [subnormal]: https://en.wikipedia.org/wiki/Denormal_number
    #[inline]
    #[must_use]
    #[unstable(feature = "f16", issue = "116909")]
    #[rustc_const_unstable(feature = "const_float_classify", issue = "72505")]
    pub const fn is_subnormal(self) -> bool {
        matches!(self.classify(), FpCategory::Subnormal)
    }

    /// Returns `true` if the number is neither zero, infinite,
    /// [subnormal], or NaN.
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// let min = f16::MIN_POSITIVE; // 2.2250738585072014e-308f16
    /// let max = f16::MAX;
    /// let lower_than_min = 1.0e-308_f16;
    /// let zero = 0.0f16;
    ///
    /// assert!(min.is_normal());
    /// assert!(max.is_normal());
    ///
    /// assert!(!zero.is_normal());
    /// assert!(!f16::NAN.is_normal());
    /// assert!(!f16::INFINITY.is_normal());
    /// // Values between `0` and `min` are Subnormal.
    /// assert!(!lower_than_min.is_normal());
    /// ```
    /// [subnormal]: https://en.wikipedia.org/wiki/Denormal_number
    #[inline]
    #[must_use]
    #[unstable(feature = "f16", issue = "116909")]
    #[rustc_const_unstable(feature = "const_float_classify", issue = "72505")]
    pub const fn is_normal(self) -> bool {
        matches!(self.classify(), FpCategory::Normal)
    }

    /// Returns the floating point category of the number. If only one property
    /// is going to be tested, it is generally faster to use the specific
    /// predicate instead.
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// use std::num::FpCategory;
    ///
    /// let num = 12.4_f16;
    /// let inf = f16::INFINITY;
    ///
    /// assert_eq!(num.classify(), FpCategory::Normal);
    /// assert_eq!(inf.classify(), FpCategory::Infinite);
    /// ```
    #[inline]
    #[unstable(feature = "f16", issue = "116909")]
    #[rustc_const_unstable(feature = "const_float_classify", issue = "72505")]
    pub const fn classify(self) -> FpCategory {
        // A previous implementation tried to only use bitmask-based checks,
        // using f16::to_bits to transmute the float to its bit repr and match on that.
        // Unfortunately, floating point numbers can be much worse than that.
        // This also needs to not result in recursive evaluations of f16::to_bits.
        //
        // On some processors, in some cases, LLVM will "helpfully" lower floating point ops,
        // in spite of a request for them using f32 and f16, to things like x87 operations.
        // These have an f16's mantissa, but can have a larger than normal exponent.
        // FIXME(jubilee): Using x87 operations is never necessary in order to function
        // on x86 processors for Rust-to-Rust calls, so this issue should not happen.
        // Code generation should be adjusted to use non-C calling conventions, avoiding this.
        //
        // Thus, a value may compare unequal to infinity, despite having a "full" exponent mask.
        // And it may not be NaN, as it can simply be an "overextended" finite value.
        if self.is_nan() {
            FpCategory::Nan
        } else {
            // However, std can't simply compare to zero to check for zero, either,
            // as correctness requires avoiding equality tests that may be Subnormal == -0.0
            // because it may be wrong under "denormals are zero" and "flush to zero" modes.
            // Most of std's targets don't use those, but they are used for thumbv7neon.
            // So, this does use bitpattern matching for the rest.

            // SAFETY: f16 to u16 is fine. Usually.
            // If control flow has gotten this far, the value is definitely in one of the categories
            // that f16::partial_classify can correctly analyze.
            unsafe { f16::partial_classify(self) }
        }
    }

    // This doesn't actually return a right answer for NaN on purpose,
    // seeing as how it cannot correctly discern between a floating point NaN,
    // and some normal floating point numbers truncated from an x87 FPU.
    #[inline]
    #[rustc_const_unstable(feature = "const_float_classify", issue = "72505")]
    const unsafe fn partial_classify(self) -> FpCategory {
        // SAFETY: The caller is not asking questions for which this will tell lies.
        let b = unsafe { mem::transmute::<f16, u16>(self) };
        match (b & Self::MAN_MASK, b & Self::EXP_MASK) {
            (0, Self::EXP_MASK) => FpCategory::Infinite,
            (0, 0) => FpCategory::Zero,
            (_, 0) => FpCategory::Subnormal,
            _ => FpCategory::Normal,
        }
    }

    // This operates on bits, and only bits, so it can ignore concerns about weird FPUs.
    // FIXME(jubilee): In a just world, this would be the entire impl for classify,
    // plus a transmute. We do not live in a just world, but we can make it more so.
    #[inline]
    #[rustc_const_unstable(feature = "const_float_classify", issue = "72505")]
    const fn classify_bits(b: u16) -> FpCategory {
        match (b & Self::MAN_MASK, b & Self::EXP_MASK) {
            (0, Self::EXP_MASK) => FpCategory::Infinite,
            (_, Self::EXP_MASK) => FpCategory::Nan,
            (0, 0) => FpCategory::Zero,
            (_, 0) => FpCategory::Subnormal,
            _ => FpCategory::Normal,
        }
    }

    /// Returns `true` if `self` has a positive sign, including `+0.0`, NaNs with
    /// positive sign bit and positive infinity. Note that IEEE 754 doesn't assign any
    /// meaning to the sign bit in case of a NaN, and as Rust doesn't guarantee that
    /// the bit pattern of NaNs are conserved over arithmetic operations, the result of
    /// `is_sign_positive` on a NaN might produce an unexpected result in some cases.
    /// See [explanation of NaN as a special value](f32) for more info.
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// let f = 7.0_f16;
    /// let g = -7.0_f16;
    ///
    /// assert!(f.is_sign_positive());
    /// assert!(!g.is_sign_positive());
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "f16", issue = "116909")]
    #[rustc_const_unstable(feature = "const_float_classify", issue = "72505")]
    pub const fn is_sign_positive(self) -> bool {
        !self.is_sign_negative()
    }

    #[inline]
    #[must_use]
    #[doc(hidden)]
    #[unstable(feature = "f16", issue = "116909")]
    #[deprecated(since = "1.0.0", note = "renamed to is_sign_positive")]
    pub fn is_positive(self) -> bool {
        self.is_sign_positive()
    }

    /// Returns `true` if `self` has a negative sign, including `-0.0`, NaNs with
    /// negative sign bit and negative infinity. Note that IEEE 754 doesn't assign any
    /// meaning to the sign bit in case of a NaN, and as Rust doesn't guarantee that
    /// the bit pattern of NaNs are conserved over arithmetic operations, the result of
    /// `is_sign_negative` on a NaN might produce an unexpected result in some cases.
    /// See [explanation of NaN as a special value](f32) for more info.
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// let f = 7.0_f16;
    /// let g = -7.0_f16;
    ///
    /// assert!(!f.is_sign_negative());
    /// assert!(g.is_sign_negative());
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "f16", issue = "116909")]
    #[rustc_const_unstable(feature = "const_float_classify", issue = "72505")]
    pub const fn is_sign_negative(self) -> bool {
        // IEEE754 says: isSignMinus(x) is true if and only if x has negative sign. isSignMinus
        // applies to zeros and NaNs as well.
        // SAFETY: This is just transmuting to get the sign bit, it's fine.
        unsafe { mem::transmute::<f16, u16>(self) & Self::SIGN_MASK != 0 }
    }

    #[inline]
    #[must_use]
    #[doc(hidden)]
    #[unstable(feature = "f16", issue = "116909")]
    #[deprecated(since = "1.0.0", note = "renamed to is_sign_negative")]
    pub fn is_negative(self) -> bool {
        self.is_sign_negative()
    }

    /// Returns the least number greater than `self`.
    ///
    /// Let `TINY` be the smallest representable positive `f16`. Then,
    ///  - if `self.is_nan()`, this returns `self`;
    ///  - if `self` is [`NEG_INFINITY`], this returns [`MIN`];
    ///  - if `self` is `-TINY`, this returns -0.0;
    ///  - if `self` is -0.0 or +0.0, this returns `TINY`;
    ///  - if `self` is [`MAX`] or [`INFINITY`], this returns [`INFINITY`];
    ///  - otherwise the unique least value greater than `self` is returned.
    ///
    /// The identity `x.next_up() == -(-x).next_down()` holds for all non-NaN `x`. When `x`
    /// is finite `x == x.next_up().next_down()` also holds.
    ///
    /// ```rust
    /// #![feature(f16)]
    /// #![feature(float_next_up_down)]
    ///
    /// // f16::EPSILON is the difference between 1.0 and the next number up.
    /// assert_eq!(1.0f16.next_up(), 1.0 + f16::EPSILON);
    /// // But not for most numbers.
    /// assert!(0.1f16.next_up() < 0.1 + f16::EPSILON);
    /// assert_eq!(9007199254740992f16.next_up(), 9007199254740994.0);
    /// ```
    ///
    /// [`NEG_INFINITY`]: Self::NEG_INFINITY
    /// [`INFINITY`]: Self::INFINITY
    /// [`MIN`]: Self::MIN
    /// [`MAX`]: Self::MAX
    #[inline]
    #[unstable(feature = "float_next_up_down", issue = "91399")]
    #[rustc_const_unstable(feature = "float_next_up_down", issue = "91399")]
    pub const fn next_up(self) -> Self {
        // We must use strictly integer arithmetic to prevent denormals from
        // flushing to zero after an arithmetic operation on some platforms.
        let bits = self.to_bits();
        if self.is_nan() || bits == Self::INFINITY.to_bits() {
            return self;
        }

        let abs = bits & Self::CLEAR_SIGN_MASK;
        let next_bits = if abs == 0 {
            Self::TINY_BITS
        } else if bits == abs {
            bits + 1
        } else {
            bits - 1
        };
        Self::from_bits(next_bits)
    }

    /// Returns the greatest number less than `self`.
    ///
    /// Let `TINY` be the smallest representable positive `f16`. Then,
    ///  - if `self.is_nan()`, this returns `self`;
    ///  - if `self` is [`INFINITY`], this returns [`MAX`];
    ///  - if `self` is `TINY`, this returns 0.0;
    ///  - if `self` is -0.0 or +0.0, this returns `-TINY`;
    ///  - if `self` is [`MIN`] or [`NEG_INFINITY`], this returns [`NEG_INFINITY`];
    ///  - otherwise the unique greatest value less than `self` is returned.
    ///
    /// The identity `x.next_down() == -(-x).next_up()` holds for all non-NaN `x`. When `x`
    /// is finite `x == x.next_down().next_up()` also holds.
    ///
    /// ```rust
    /// #![feature(f16)]
    /// #![feature(float_next_up_down)]
    ///
    /// let x = 1.0f16;
    /// // Clamp value into range [0, 1).
    /// let clamped = x.clamp(0.0, 1.0f16.next_down());
    /// assert!(clamped < 1.0);
    /// assert_eq!(clamped.next_up(), 1.0);
    /// ```
    ///
    /// [`NEG_INFINITY`]: Self::NEG_INFINITY
    /// [`INFINITY`]: Self::INFINITY
    /// [`MIN`]: Self::MIN
    /// [`MAX`]: Self::MAX
    #[inline]
    #[unstable(feature = "float_next_up_down", issue = "91399")]
    #[rustc_const_unstable(feature = "float_next_up_down", issue = "91399")]
    pub const fn next_down(self) -> Self {
        // We must use strictly integer arithmetic to prevent denormals from
        // flushing to zero after an arithmetic operation on some platforms.
        let bits = self.to_bits();
        if self.is_nan() || bits == Self::NEG_INFINITY.to_bits() {
            return self;
        }

        let abs = bits & Self::CLEAR_SIGN_MASK;
        let next_bits = if abs == 0 {
            Self::NEG_TINY_BITS
        } else if bits == abs {
            bits - 1
        } else {
            bits + 1
        };
        Self::from_bits(next_bits)
    }

    /// Takes the reciprocal (inverse) of a number, `1/x`.
    ///
    /// ```
    /// #![feature(f16)]
    /// #![feature(f16_math)]
    ///
    /// let x = 2.0_f16;
    /// let abs_difference = (x.recip() - (1.0 / x)).abs();
    ///
    /// assert!(abs_difference < 1e-6);
    /// ```
    #[inline]
    #[unstable(feature = "f16", issue = "116909")]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub fn recip(self) -> f16 {
        1.0 / self
    }

    /// Converts radians to degrees.
    ///
    /// ```
    /// #![feature(f16)]
    /// #![feature(f16_math)]
    ///
    /// let angle = std::f16::consts::PI;
    ///
    /// let abs_difference = (angle.to_degrees() - 180.0).abs();
    ///
    /// // `f16` loses precision pretty fast
    /// assert!(abs_difference < 1e-6);
    /// ```
    #[inline]
    #[unstable(feature = "f16", issue = "116909")]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub fn to_degrees(self) -> f16 {
        // The division here is correctly rounded with respect to the true
        // value of 180/π. (This differs from f32, where a constant must be
        // used to ensure a correctly rounded result.)
        self * (180.0f16 / consts::PI)
    }

    /// Converts degrees to radians.
    ///
    /// ```
    /// #![feature(f16)]
    /// #![feature(f16_math)]
    ///
    /// let angle = 180.0_f16;
    ///
    /// let abs_difference = (angle.to_radians() - std::f16::consts::PI).abs();
    ///
    /// assert!(abs_difference < 0.01);
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    #[unstable(feature = "f16", issue = "116909")]
    pub fn to_radians(self) -> f16 {
        let value: f16 = consts::PI;
        self * (value / 180.0)
    }

    /// Returns the maximum of the two numbers, ignoring NaN.
    ///
    /// If one of the arguments is NaN, then the other argument is returned.
    /// This follows the IEEE 754-2008 semantics for maxNum, except for handling of signaling NaNs;
    /// this function handles all NaNs the same way and avoids maxNum's problems with associativity.
    /// This also matches the behavior of libm’s fmax.
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// let x = 1.0_f16;
    /// let y = 2.0_f16;
    ///
    /// assert_eq!(x.max(y), y);
    /// ```
    #[inline]
    #[must_use = "this returns the result of the comparison, without modifying either input"]
    #[unstable(feature = "f16", issue = "116909")]
    pub fn max(self, other: f16) -> f16 {
        intrinsics::maxnumf16(self, other)
    }

    /// Returns the minimum of the two numbers, ignoring NaN.
    ///
    /// If one of the arguments is NaN, then the other argument is returned.
    /// This follows the IEEE 754-2008 semantics for minNum, except for handling of signaling NaNs;
    /// this function handles all NaNs the same way and avoids minNum's problems with associativity.
    /// This also matches the behavior of libm’s fmin.
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// let x = 1.0_f16;
    /// let y = 2.0_f16;
    ///
    /// assert_eq!(x.min(y), x);
    /// ```
    #[inline]
    #[must_use = "this returns the result of the comparison, without modifying either input"]
    #[unstable(feature = "f16", issue = "116909")]
    pub fn min(self, other: f16) -> f16 {
        intrinsics::minnumf16(self, other)
    }

    /// Returns the maximum of the two numbers, propagating NaN.
    ///
    /// This returns NaN when *either* argument is NaN, as opposed to
    /// [`f16::max`] which only returns NaN when *both* arguments are NaN.
    ///
    /// ```
    /// #![feature(f16)]
    /// #![feature(float_minimum_maximum)]
    ///
    /// let x = 1.0_f16;
    /// let y = 2.0_f16;
    ///
    /// assert_eq!(x.maximum(y), y);
    /// assert!(x.maximum(f16::NAN).is_nan());
    /// ```
    ///
    /// If one of the arguments is NaN, then NaN is returned. Otherwise this returns the greater
    /// of the two numbers. For this operation, -0.0 is considered to be less than +0.0.
    /// Note that this follows the semantics specified in IEEE 754-2019.
    ///
    /// Also note that "propagation" of NaNs here doesn't necessarily mean that the bitpattern of a NaN
    /// operand is conserved; see [explanation of NaN as a special value](f32) for more info.
    #[inline]
    #[must_use = "this returns the result of the comparison, without modifying either input"]
    #[unstable(feature = "float_minimum_maximum", issue = "91079")]
    pub fn maximum(self, other: f16) -> f16 {
        if self > other {
            self
        } else if other > self {
            other
        } else if self == other {
            if self.is_sign_positive() && other.is_sign_negative() { self } else { other }
        } else {
            self + other
        }
    }

    /// Returns the minimum of the two numbers, propagating NaN.
    ///
    /// This returns NaN when *either* argument is NaN, as opposed to
    /// [`f16::min`] which only returns NaN when *both* arguments are NaN.
    ///
    /// ```
    /// #![feature(f16)]
    /// #![feature(float_minimum_maximum)]
    ///
    /// let x = 1.0_f16;
    /// let y = 2.0_f16;
    ///
    /// assert_eq!(x.minimum(y), x);
    /// assert!(x.minimum(f16::NAN).is_nan());
    /// ```
    ///
    /// If one of the arguments is NaN, then NaN is returned. Otherwise this returns the lesser
    /// of the two numbers. For this operation, -0.0 is considered to be less than +0.0.
    /// Note that this follows the semantics specified in IEEE 754-2019.
    ///
    /// Also note that "propagation" of NaNs here doesn't necessarily mean that the bitpattern of a NaN
    /// operand is conserved; see [explanation of NaN as a special value](f32) for more info.
    #[inline]
    #[must_use = "this returns the result of the comparison, without modifying either input"]
    #[unstable(feature = "float_minimum_maximum", issue = "91079")]
    pub fn minimum(self, other: f16) -> f16 {
        if self < other {
            self
        } else if other < self {
            other
        } else if self == other {
            if self.is_sign_negative() && other.is_sign_positive() { self } else { other }
        } else {
            self + other
        }
    }

    /// Calculates the middle point of `self` and `rhs`.
    ///
    /// This returns NaN when *either* argument is NaN or if a combination of
    /// +inf and -inf is provided as arguments.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f16)]
    /// #![feature(num_midpoint)]
    ///
    /// assert_eq!(1f16.midpoint(4.0), 2.5);
    /// assert_eq!((-5.5f16).midpoint(8.0), 1.25);
    /// ```
    #[inline]
    #[unstable(feature = "num_midpoint", issue = "110840")]
    pub fn midpoint(self, other: f16) -> f16 {
        const LO: f16 = f16::MIN_POSITIVE * 2.;
        const HI: f16 = f16::MAX / 2.;

        let (a, b) = (self, other);
        let abs_a = a.abs_private();
        let abs_b = b.abs_private();

        if abs_a <= HI && abs_b <= HI {
            // Overflow is impossible
            (a + b) / 2.
        } else if abs_a < LO {
            // Not safe to halve a
            a + (b / 2.)
        } else if abs_b < LO {
            // Not safe to halve b
            (a / 2.) + b
        } else {
            // Not safe to halve a and b
            (a / 2.) + (b / 2.)
        }
    }

    /// Rounds toward zero and converts to any primitive integer type,
    /// assuming that the value is finite and fits in that type.
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// let value = 4.6_f16;
    /// let rounded = unsafe { value.to_int_unchecked::<u16>() };
    /// assert_eq!(rounded, 4);
    ///
    /// let value = -128.9_f16;
    /// let rounded = unsafe { value.to_int_unchecked::<i8>() };
    /// assert_eq!(rounded, i8::MIN);
    /// ```
    ///
    /// # Safety
    ///
    /// The value must:
    ///
    /// * Not be `NaN`
    /// * Not be infinite
    /// * Be representable in the return type `Int`, after truncating off its fractional part
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    #[unstable(feature = "f16", issue = "116909")]
    pub unsafe fn to_int_unchecked<Int>(self) -> Int
    where
        Self: FloatToInt<Int>,
    {
        // SAFETY: the caller must uphold the safety contract for
        // `FloatToInt::to_int_unchecked`.
        unsafe { FloatToInt::<Int>::to_int_unchecked(self) }
    }

    /// Raw transmutation to `u16`.
    ///
    /// This is currently identical to `transmute::<f16, u16>(self)` on all platforms.
    ///
    /// See [`from_bits`](Self::from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// Note that this function is distinct from `as` casting, which attempts to
    /// preserve the *numeric* value, and not the bitwise value.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// assert!((1f16).to_bits() != 1f16 as u16); // to_bits() is not casting!
    /// assert_eq!((12.5f16).to_bits(), 0x4A40);
    ///
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    #[unstable(feature = "f16", issue = "116909")]
    #[rustc_const_unstable(feature = "const_float_bits_conv", issue = "72447")]
    pub const fn to_bits(self) -> u16 {
        // SAFETY: `u16` is a plain old datatype so we can always transmute to it.
        // ...sorta.
        //
        // See the SAFETY comment in f16::from_bits for more.
        #[inline]
        #[rustc_const_unstable(feature = "const_float_bits_conv", issue = "72447")]
        const fn ct_f16_to_u16(ct: f16) -> u16 {
            match ct.classify() {
                FpCategory::Nan => {
                    panic!("const-eval error: cannot use f16::to_bits on a NaN")
                }
                FpCategory::Subnormal => {
                    panic!("const-eval error: cannot use f16::to_bits on a subnormal number")
                }
                FpCategory::Infinite | FpCategory::Normal | FpCategory::Zero => {
                    // SAFETY: We have a normal floating point number. Now we transmute, i.e. do a bitcopy.
                    unsafe { mem::transmute::<f16, u16>(ct) }
                }
            }
        }

        #[inline(always)] // See https://github.com/rust-lang/compiler-builtins/issues/491
        fn rt_f16_to_u16(rt: f16) -> u16 {
            // SAFETY: `u16` is a plain old datatype so we can always... uh...
            // ...look, just pretend you forgot what you just read.
            // Stability concerns.
            unsafe { mem::transmute::<f16, u16>(rt) }
        }
        // SAFETY: We use internal implementations that either always work or fail at compile time.
        unsafe { intrinsics::const_eval_select((self,), ct_f16_to_u16, rt_f16_to_u16) }
    }

    /// Raw transmutation from `u16`.
    ///
    /// This is currently identical to `transmute::<u16, f16>(v)` on all platforms.
    /// It turns out this is incredibly portable, for two reasons:
    ///
    /// * Floats and Ints have the same endianness on all supported platforms.
    /// * IEEE 754 very precisely specifies the bit layout of floats.
    ///
    /// However there is one caveat: prior to the 2008 version of IEEE 754, how
    /// to interpret the NaN signaling bit wasn't actually specified. Most platforms
    /// (notably x86 and ARM) picked the interpretation that was ultimately
    /// standardized in 2008, but some didn't (notably MIPS). As a result, all
    /// signaling NaNs on MIPS are quiet NaNs on x86, and vice-versa.
    ///
    /// Rather than trying to preserve signaling-ness cross-platform, this
    /// implementation favors preserving the exact bits. This means that
    /// any payloads encoded in NaNs will be preserved even if the result of
    /// this method is sent over the network from an x86 machine to a MIPS one.
    ///
    /// If the results of this method are only manipulated by the same
    /// architecture that produced them, then there is no portability concern.
    ///
    /// If the input isn't NaN, then there is no portability concern.
    ///
    /// If you don't care about signaling-ness (very likely), then there is no
    /// portability concern.
    ///
    /// Note that this function is distinct from `as` casting, which attempts to
    /// preserve the *numeric* value, and not the bitwise value.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// let v = f16::from_bits(0x4a40);
    /// assert_eq!(v, 12.5);
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "f16", issue = "116909")]
    #[rustc_const_unstable(feature = "const_float_bits_conv", issue = "72447")]
    pub const fn from_bits(v: u16) -> Self {
        // It turns out the safety issues with sNaN were overblown! Hooray!
        // SAFETY: `u16` is a plain old datatype so we can always transmute from it
        // ...sorta.
        //
        // It turns out that at runtime, it is possible for a floating point number
        // to be subject to floating point modes that alter nonzero subnormal numbers
        // to zero on reads and writes, aka "denormals are zero" and "flush to zero".
        // This is not a problem usually, but at least one tier2 platform for Rust
        // actually exhibits an FTZ behavior by default: thumbv7neon
        // aka "the Neon FPU in AArch32 state"
        //
        // Even with this, not all instructions exhibit the FTZ behaviors on thumbv7neon,
        // so this should load the same bits if LLVM emits the "correct" instructions,
        // but LLVM sometimes makes interesting choices about float optimization,
        // and other FPUs may do similar. Thus, it is wise to indulge luxuriously in caution.
        //
        // In addition, on x86 targets with SSE or SSE2 disabled and the x87 FPU enabled,
        // i.e. not soft-float, the way Rust does parameter passing can actually alter
        // a number that is "not infinity" to have the same exponent as infinity,
        // in a slightly unpredictable manner.
        //
        // And, of course evaluating to a NaN value is fairly nondeterministic.
        // More precisely: when NaN should be returned is knowable, but which NaN?
        // So far that's defined by a combination of LLVM and the CPU, not Rust.
        // This function, however, allows observing the bitstring of a NaN,
        // thus introspection on CTFE.
        //
        // In order to preserve, at least for the moment, const-to-runtime equivalence,
        // reject any of these possible situations from happening.
        #[inline]
        #[rustc_const_unstable(feature = "const_float_bits_conv", issue = "72447")]
        const fn ct_u16_to_f16(ct: u16) -> f16 {
            match f16::classify_bits(ct) {
                FpCategory::Subnormal => {
                    panic!("const-eval error: cannot use f16::from_bits on a subnormal number")
                }
                FpCategory::Nan => {
                    panic!("const-eval error: cannot use f16::from_bits on NaN")
                }
                FpCategory::Infinite | FpCategory::Normal | FpCategory::Zero => {
                    // SAFETY: It's not a frumious number
                    unsafe { mem::transmute::<u16, f16>(ct) }
                }
            }
        }

        #[inline(always)] // See https://github.com/rust-lang/compiler-builtins/issues/491
        fn rt_u16_to_f16(rt: u16) -> f16 {
            // SAFETY: `u16` is a plain old datatype so we can always... uh...
            // ...look, just pretend you forgot what you just read.
            // Stability concerns.
            unsafe { mem::transmute::<u16, f16>(rt) }
        }
        // SAFETY: We use internal implementations that either always work or fail at compile time.
        unsafe { intrinsics::const_eval_select((v,), ct_u16_to_f16, rt_u16_to_f16) }
    }

    /// Return the memory representation of this floating point number as a byte array in
    /// big-endian (network) byte order.
    ///
    /// See [`from_bits`](Self::from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// let bytes = 12.5f16.to_be_bytes();
    /// assert_eq!(bytes, [0x4a, 0x40]);
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    #[unstable(feature = "f16", issue = "116909")]
    #[rustc_const_unstable(feature = "const_float_bits_conv", issue = "72447")]
    pub const fn to_be_bytes(self) -> [u8; 2] {
        self.to_bits().to_be_bytes()
    }

    /// Return the memory representation of this floating point number as a byte array in
    /// little-endian byte order.
    ///
    /// See [`from_bits`](Self::from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// let bytes = 12.5f16.to_le_bytes();
    /// assert_eq!(bytes, [0x40, 0x4a]);
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    #[unstable(feature = "f16", issue = "116909")]
    #[rustc_const_unstable(feature = "const_float_bits_conv", issue = "72447")]
    pub const fn to_le_bytes(self) -> [u8; 2] {
        self.to_bits().to_le_bytes()
    }

    /// Return the memory representation of this floating point number as a byte array in
    /// native byte order.
    ///
    /// As the target platform's native endianness is used, portable code
    /// should use [`to_be_bytes`] or [`to_le_bytes`], as appropriate, instead.
    ///
    /// [`to_be_bytes`]: f16::to_be_bytes
    /// [`to_le_bytes`]: f16::to_le_bytes
    ///
    /// See [`from_bits`](Self::from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// let bytes = 12.5f16.to_ne_bytes();
    /// assert_eq!(
    ///     bytes,
    ///     if cfg!(target_endian = "big") {
    ///         [0x4a, 0x40]
    ///     } else {
    ///         [0x40, 0x4a]
    ///     }
    /// );
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    #[unstable(feature = "f16", issue = "116909")]
    #[rustc_const_unstable(feature = "const_float_bits_conv", issue = "72447")]
    pub const fn to_ne_bytes(self) -> [u8; 2] {
        self.to_bits().to_ne_bytes()
    }

    /// Create a floating point value from its representation as a byte array in big endian.
    ///
    /// See [`from_bits`](Self::from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// let value = f16::from_be_bytes([0x4a, 0x40]);
    /// assert_eq!(value, 12.5);
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "f16", issue = "116909")]
    #[rustc_const_unstable(feature = "const_float_bits_conv", issue = "72447")]
    pub const fn from_be_bytes(bytes: [u8; 2]) -> Self {
        Self::from_bits(u16::from_be_bytes(bytes))
    }

    /// Create a floating point value from its representation as a byte array in little endian.
    ///
    /// See [`from_bits`](Self::from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// let value = f16::from_le_bytes([0x40, 0x4a]);
    /// assert_eq!(value, 12.5);
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "f16", issue = "116909")]
    #[rustc_const_unstable(feature = "const_float_bits_conv", issue = "72447")]
    pub const fn from_le_bytes(bytes: [u8; 2]) -> Self {
        Self::from_bits(u16::from_le_bytes(bytes))
    }

    /// Create a floating point value from its representation as a byte array in native endian.
    ///
    /// As the target platform's native endianness is used, portable code
    /// likely wants to use [`from_be_bytes`] or [`from_le_bytes`], as
    /// appropriate instead.
    ///
    /// [`from_be_bytes`]: f16::from_be_bytes
    /// [`from_le_bytes`]: f16::from_le_bytes
    ///
    /// See [`from_bits`](Self::from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// let value = f16::from_ne_bytes(if cfg!(target_endian = "big") {
    ///     [0x4a, 0x40]
    /// } else {
    ///     [0x40, 0x4a]
    /// });
    /// assert_eq!(value, 12.5);
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "f16", issue = "116909")]
    #[rustc_const_unstable(feature = "const_float_bits_conv", issue = "72447")]
    pub const fn from_ne_bytes(bytes: [u8; 2]) -> Self {
        Self::from_bits(u16::from_ne_bytes(bytes))
    }

    /// Return the ordering between `self` and `other`.
    ///
    /// Unlike the standard partial comparison between floating point numbers,
    /// this comparison always produces an ordering in accordance to
    /// the `totalOrder` predicate as defined in the IEEE 754 (2008 revision)
    /// floating point standard. The values are ordered in the following sequence:
    ///
    /// - negative quiet NaN
    /// - negative signaling NaN
    /// - negative infinity
    /// - negative numbers
    /// - negative subnormal numbers
    /// - negative zero
    /// - positive zero
    /// - positive subnormal numbers
    /// - positive numbers
    /// - positive infinity
    /// - positive signaling NaN
    /// - positive quiet NaN.
    ///
    /// The ordering established by this function does not always agree with the
    /// [`PartialOrd`] and [`PartialEq`] implementations of `f16`. For example,
    /// they consider negative and positive zero equal, while `total_cmp`
    /// doesn't.
    ///
    /// The interpretation of the signaling NaN bit follows the definition in
    /// the IEEE 754 standard, which may not match the interpretation by some of
    /// the older, non-conformant (e.g. MIPS) hardware implementations.
    ///
    /// # Example
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// struct GoodBoy {
    ///     name: String,
    ///     weight: f16,
    /// }
    ///
    /// let mut bois = vec![
    ///     GoodBoy { name: "Pucci".to_owned(), weight: 0.1 },
    ///     GoodBoy { name: "Woofer".to_owned(), weight: 99.0 },
    ///     GoodBoy { name: "Yapper".to_owned(), weight: 10.0 },
    ///     GoodBoy { name: "Chonk".to_owned(), weight: f16::INFINITY },
    ///     GoodBoy { name: "Abs. Unit".to_owned(), weight: f16::NAN },
    ///     GoodBoy { name: "Floaty".to_owned(), weight: -5.0 },
    /// ];
    ///
    /// bois.sort_by(|a, b| a.weight.total_cmp(&b.weight));
    /// # bois.into_iter().map(|b| b.weight)
    /// #     .zip([-5.0, 0.1, 10.0, 99.0, f16::INFINITY, f16::NAN].iter())
    /// #     .for_each(|(a, b)| assert_eq!(a.to_bits(), b.to_bits(), "{a} != {b}"))
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "f16", issue = "116909")]
    pub fn total_cmp(&self, other: &Self) -> crate::cmp::Ordering {
        let mut left = self.to_bits() as i16;
        let mut right = other.to_bits() as i16;

        // In case of negatives, flip all the bits except the sign
        // to achieve a similar layout as two's complement integers
        //
        // Why does this work? IEEE 754 floats consist of three fields:
        // Sign bit, exponent and mantissa. The set of exponent and mantissa
        // fields as a whole have the property that their bitwise order is
        // equal to the numeric magnitude where the magnitude is defined.
        // The magnitude is not normally defined on NaN values, but
        // IEEE 754 totalOrder defines the NaN values also to follow the
        // bitwise order. This leads to order explained in the doc comment.
        // However, the representation of magnitude is the same for negative
        // and positive numbers – only the sign bit is different.
        // To easily compare the floats as signed integers, we need to
        // flip the exponent and mantissa bits in case of negative numbers.
        // We effectively convert the numbers to "two's complement" form.
        //
        // To do the flipping, we construct a mask and XOR against it.
        // We branchlessly calculate an "all-ones except for the sign bit"
        // mask from negative-signed values: right shifting sign-extends
        // the integer, so we "fill" the mask with sign bits, and then
        // convert to unsigned to push one more zero bit.
        // On positive values, the mask is all zeros, so it's a no-op.
        left ^= (((left >> 15) as u16) >> 1) as i16;
        right ^= (((right >> 15) as u16) >> 1) as i16;

        left.cmp(&right)
    }

    /// Restrict a value to a certain interval unless it is NaN.
    ///
    /// Returns `max` if `self` is greater than `max`, and `min` if `self` is
    /// less than `min`. Otherwise this returns `self`.
    ///
    /// Note that this function returns NaN if the initial value was NaN as
    /// well.
    ///
    /// # Panics
    ///
    /// Panics if `min > max`, `min` is NaN, or `max` is NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f16)]
    ///
    /// assert!((-3.0f16).clamp(-2.0, 1.0) == -2.0);
    /// assert!((0.0f16).clamp(-2.0, 1.0) == 0.0);
    /// assert!((2.0f16).clamp(-2.0, 1.0) == 1.0);
    /// assert!((f16::NAN).clamp(-2.0, 1.0).is_nan());
    /// ```
    #[inline]
    #[unstable(feature = "f16", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn clamp(mut self, min: f16, max: f16) -> f16 {
        assert!(min <= max, "min > max, or either was NaN. min = {min:?}, max = {max:?}");
        if self < min {
            self = min;
        }
        if self > max {
            self = max;
        }
        self
    }
}
