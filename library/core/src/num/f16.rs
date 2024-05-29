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

use crate::mem;

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

    /// The golden ratio (φ)
    #[unstable(feature = "f16", issue = "116909")]
    // Also, #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const PHI: f16 = 1.618033988749894848204586834365638118_f16;

    /// The Euler-Mascheroni constant (γ)
    #[unstable(feature = "f16", issue = "116909")]
    // Also, #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const EGAMMA: f16 = 0.577215664901532860606512090082402431_f16;

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

    /// 1/sqrt(π)
    #[unstable(feature = "f16", issue = "116909")]
    // Also, #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const FRAC_1_SQRT_PI: f16 = 0.564189583547756286948079451560772586_f16;

    /// 1/sqrt(2π)
    #[doc(alias = "FRAC_1_SQRT_TAU")]
    #[unstable(feature = "f16", issue = "116909")]
    // Also, #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const FRAC_1_SQRT_2PI: f16 = 0.398942280401432677939946059934381868_f16;

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

    /// sqrt(3)
    #[unstable(feature = "f16", issue = "116909")]
    // Also, #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const SQRT_3: f16 = 1.732050807568877293527446341505872367_f16;

    /// 1/sqrt(3)
    #[unstable(feature = "f16", issue = "116909")]
    // Also, #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const FRAC_1_SQRT_3: f16 = 0.577350269189625764509148780501957456_f16;

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
    // FIXME(f16_f128): almost all methods in this `impl` are missing examples and a const
    // implementation. Add these once we can run code on all platforms and have f16/f128 in CTFE.

    /// The radix or base of the internal representation of `f16`.
    #[unstable(feature = "f16", issue = "116909")]
    pub const RADIX: u32 = 2;

    /// Number of significant digits in base 2.
    #[unstable(feature = "f16", issue = "116909")]
    pub const MANTISSA_DIGITS: u32 = 11;

    /// Approximate number of significant digits in base 10.
    ///
    /// This is the maximum <i>x</i> such that any decimal number with <i>x</i>
    /// significant digits can be converted to `f16` and back without loss.
    ///
    /// Equal to floor(log<sub>10</sub>&nbsp;2<sup>[`MANTISSA_DIGITS`]&nbsp;&minus;&nbsp;1</sup>).
    ///
    /// [`MANTISSA_DIGITS`]: f16::MANTISSA_DIGITS
    #[unstable(feature = "f16", issue = "116909")]
    pub const DIGITS: u32 = 3;

    /// [Machine epsilon] value for `f16`.
    ///
    /// This is the difference between `1.0` and the next larger representable number.
    ///
    /// Equal to 2<sup>1&nbsp;&minus;&nbsp;[`MANTISSA_DIGITS`]</sup>.
    ///
    /// [Machine epsilon]: https://en.wikipedia.org/wiki/Machine_epsilon
    /// [`MANTISSA_DIGITS`]: f16::MANTISSA_DIGITS
    #[unstable(feature = "f16", issue = "116909")]
    pub const EPSILON: f16 = 9.7656e-4_f16;

    /// Smallest finite `f16` value.
    ///
    /// Equal to &minus;[`MAX`].
    ///
    /// [`MAX`]: f16::MAX
    #[unstable(feature = "f16", issue = "116909")]
    pub const MIN: f16 = -6.5504e+4_f16;
    /// Smallest positive normal `f16` value.
    ///
    /// Equal to 2<sup>[`MIN_EXP`]&nbsp;&minus;&nbsp;1</sup>.
    ///
    /// [`MIN_EXP`]: f16::MIN_EXP
    #[unstable(feature = "f16", issue = "116909")]
    pub const MIN_POSITIVE: f16 = 6.1035e-5_f16;
    /// Largest finite `f16` value.
    ///
    /// Equal to
    /// (1&nbsp;&minus;&nbsp;2<sup>&minus;[`MANTISSA_DIGITS`]</sup>)&nbsp;2<sup>[`MAX_EXP`]</sup>.
    ///
    /// [`MANTISSA_DIGITS`]: f16::MANTISSA_DIGITS
    /// [`MAX_EXP`]: f16::MAX_EXP
    #[unstable(feature = "f16", issue = "116909")]
    pub const MAX: f16 = 6.5504e+4_f16;

    /// One greater than the minimum possible normal power of 2 exponent.
    ///
    /// If <i>x</i>&nbsp;=&nbsp;`MIN_EXP`, then normal numbers
    /// ≥&nbsp;0.5&nbsp;×&nbsp;2<sup><i>x</i></sup>.
    #[unstable(feature = "f16", issue = "116909")]
    pub const MIN_EXP: i32 = -13;
    /// Maximum possible power of 2 exponent.
    ///
    /// If <i>x</i>&nbsp;=&nbsp;`MAX_EXP`, then normal numbers
    /// &lt;&nbsp;1&nbsp;×&nbsp;2<sup><i>x</i></sup>.
    #[unstable(feature = "f16", issue = "116909")]
    pub const MAX_EXP: i32 = 16;

    /// Minimum <i>x</i> for which 10<sup><i>x</i></sup> is normal.
    ///
    /// Equal to ceil(log<sub>10</sub>&nbsp;[`MIN_POSITIVE`]).
    ///
    /// [`MIN_POSITIVE`]: f16::MIN_POSITIVE
    #[unstable(feature = "f16", issue = "116909")]
    pub const MIN_10_EXP: i32 = -4;
    /// Maximum <i>x</i> for which 10<sup><i>x</i></sup> is normal.
    ///
    /// Equal to floor(log<sub>10</sub>&nbsp;[`MAX`]).
    ///
    /// [`MAX`]: f16::MAX
    #[unstable(feature = "f16", issue = "116909")]
    pub const MAX_10_EXP: i32 = 4;

    /// Returns `true` if this value is NaN.
    #[inline]
    #[must_use]
    #[unstable(feature = "f16", issue = "116909")]
    #[allow(clippy::eq_op)] // > if you intended to check if the operand is NaN, use `.is_nan()` instead :)
    pub const fn is_nan(self) -> bool {
        self != self
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
    pub fn is_sign_positive(self) -> bool {
        !self.is_sign_negative()
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
    pub fn is_sign_negative(self) -> bool {
        // IEEE754 says: isSignMinus(x) is true if and only if x has negative sign. isSignMinus
        // applies to zeros and NaNs as well.
        // SAFETY: This is just transmuting to get the sign bit, it's fine.
        (self.to_bits() & (1 << 15)) != 0
    }

    /// Raw transmutation to `u16`.
    ///
    /// This is currently identical to `transmute::<f16, u16>(self)` on all platforms.
    ///
    /// See [`from_bits`](#method.from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// Note that this function is distinct from `as` casting, which attempts to
    /// preserve the *numeric* value, and not the bitwise value.
    #[inline]
    #[unstable(feature = "f16", issue = "116909")]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub fn to_bits(self) -> u16 {
        // SAFETY: `u16` is a plain old datatype so we can always... uh...
        // ...look, just pretend you forgot what you just read.
        // Stability concerns.
        unsafe { mem::transmute(self) }
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
    /// If you don't care about signalingness (very likely), then there is no
    /// portability concern.
    ///
    /// Note that this function is distinct from `as` casting, which attempts to
    /// preserve the *numeric* value, and not the bitwise value.
    #[inline]
    #[must_use]
    #[unstable(feature = "f16", issue = "116909")]
    pub fn from_bits(v: u16) -> Self {
        // SAFETY: `u16` is a plain old datatype so we can always... uh...
        // ...look, just pretend you forgot what you just read.
        // Stability concerns.
        unsafe { mem::transmute(v) }
    }
}
