//! Constants for the `f128` quadruple-precision floating point type.
//!
//! *[See also the `f128` primitive type][f128].*
//!
//! Mathematically significant numbers are provided in the `consts` sub-module.
//!
//! For the constants defined directly in this module
//! (as distinct from those defined in the `consts` sub-module),
//! new code should instead use the associated constants
//! defined directly on the `f128` type.

#![unstable(feature = "f128", issue = "116909")]

use crate::mem;

/// Basic mathematical constants.
#[unstable(feature = "f128", issue = "116909")]
pub mod consts {
    /// Archimedes' constant (π)
    #[unstable(feature = "f128", issue = "116909")]
    pub const PI: f128 = 3.14159265358979323846264338327950288419716939937510582097494_f128;

    /// The full circle constant (τ)
    ///
    /// Equal to 2π.
    #[unstable(feature = "f128", issue = "116909")]
    pub const TAU: f128 = 6.28318530717958647692528676655900576839433879875021164194989_f128;

    /// The golden ratio (φ)
    #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const PHI: f128 = 1.61803398874989484820458683436563811772030917980576286213545_f128;

    /// The Euler-Mascheroni constant (γ)
    #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const EGAMMA: f128 = 0.577215664901532860606512090082402431042159335939923598805767_f128;

    /// π/2
    #[unstable(feature = "f128", issue = "116909")]
    pub const FRAC_PI_2: f128 = 1.57079632679489661923132169163975144209858469968755291048747_f128;

    /// π/3
    #[unstable(feature = "f128", issue = "116909")]
    pub const FRAC_PI_3: f128 = 1.04719755119659774615421446109316762806572313312503527365831_f128;

    /// π/4
    #[unstable(feature = "f128", issue = "116909")]
    pub const FRAC_PI_4: f128 = 0.785398163397448309615660845819875721049292349843776455243736_f128;

    /// π/6
    #[unstable(feature = "f128", issue = "116909")]
    pub const FRAC_PI_6: f128 = 0.523598775598298873077107230546583814032861566562517636829157_f128;

    /// π/8
    #[unstable(feature = "f128", issue = "116909")]
    pub const FRAC_PI_8: f128 = 0.392699081698724154807830422909937860524646174921888227621868_f128;

    /// 1/π
    #[unstable(feature = "f128", issue = "116909")]
    pub const FRAC_1_PI: f128 = 0.318309886183790671537767526745028724068919291480912897495335_f128;

    /// 1/sqrt(π)
    #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const FRAC_1_SQRT_PI: f128 =
        0.564189583547756286948079451560772585844050629328998856844085_f128;

    /// 2/π
    #[unstable(feature = "f128", issue = "116909")]
    pub const FRAC_2_PI: f128 = 0.636619772367581343075535053490057448137838582961825794990669_f128;

    /// 2/sqrt(π)
    #[unstable(feature = "f128", issue = "116909")]
    pub const FRAC_2_SQRT_PI: f128 =
        1.12837916709551257389615890312154517168810125865799771368817_f128;

    /// sqrt(2)
    #[unstable(feature = "f128", issue = "116909")]
    pub const SQRT_2: f128 = 1.41421356237309504880168872420969807856967187537694807317668_f128;

    /// 1/sqrt(2)
    #[unstable(feature = "f128", issue = "116909")]
    pub const FRAC_1_SQRT_2: f128 =
        0.70710678118654752440084436210484903928483593768847403658834_f128;

    /// sqrt(3)
    #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const SQRT_3: f128 = 1.732050807568877293527446341505872366942805253810380628055807_f128;
    /// 1/sqrt(3)
    #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const FRAC_1_SQRT_3: f128 =
        0.577350269189625764509148780501957455647601751270126876018602_f128;

    /// Euler's number (e)
    #[unstable(feature = "f128", issue = "116909")]
    pub const E: f128 = 2.71828182845904523536028747135266249775724709369995957496697_f128;

    /// log<sub>2</sub>(10)
    #[unstable(feature = "f128", issue = "116909")]
    pub const LOG2_10: f128 = 3.32192809488736234787031942948939017586483139302458061205476_f128;

    /// log<sub>2</sub>(e)
    #[unstable(feature = "f128", issue = "116909")]
    pub const LOG2_E: f128 = 1.44269504088896340735992468100189213742664595415298593413545_f128;

    /// log<sub>10</sub>(2)
    #[unstable(feature = "f128", issue = "116909")]
    pub const LOG10_2: f128 = 0.301029995663981195213738894724493026768189881462108541310427_f128;

    /// log<sub>10</sub>(e)
    #[unstable(feature = "f128", issue = "116909")]
    pub const LOG10_E: f128 = 0.434294481903251827651128918916605082294397005803666566114454_f128;

    /// ln(2)
    #[unstable(feature = "f128", issue = "116909")]
    pub const LN_2: f128 = 0.69314718055994530941723212145817656807550013436025525412068_f128;

    /// ln(10)
    #[unstable(feature = "f128", issue = "116909")]
    pub const LN_10: f128 = 2.30258509299404568401799145468436420760110148862877297603333_f128;
}

#[cfg(not(test))]
impl f128 {
    /// The radix or base of the internal representation of `f128`.
    #[unstable(feature = "f128", issue = "116909")]
    pub const RADIX: u32 = 128;

    /// Number of significant digits in base 2.
    #[unstable(feature = "f128", issue = "116909")]
    pub const MANTISSA_DIGITS: u32 = 112;

    /// Approximate number of significant digits in base 10.
    ///
    /// This is the maximum _x_ such that any decimal number with _x_
    /// significant digits can be converted to `f32` and back without loss.
    ///
    /// Equal to floor(log<sub>10</sub>&nbsp;2<sup>[`MANTISSA_DIGITS`]&nbsp;&minus;&nbsp;1</sup>).
    #[unstable(feature = "f128", issue = "116909")]
    pub const DIGITS: u32 = 33;

    /// [Machine epsilon] value for `f128`.
    ///
    /// This is the difference between `1.0` and the next larger representable number.
    ///
    /// Equal to 2<sup>1&nbsp;&minus;&nbsp;[`MANTISSA_DIGITS`]</sup>.
    ///
    /// [Machine epsilon]: https://en.wikipedia.org/wiki/Machine_epsilon
    #[unstable(feature = "f128", issue = "116909")]
    pub const EPSILON: f128 = 1.92592994438723585305597794258492732e-34_f128;

    /// Smallest finite `f128` value.
    ///
    /// Equal to &minus;[`MAX`].
    #[cfg(not(bootstrap))]
    #[unstable(feature = "f128", issue = "116909")]
    pub const MIN: f128 = -1.1897314953572317650857593266280070162e+4932_f128;

    /// Smallest positive normal `f128` value.
    ///
    /// Equal to 2<sup>[`MIN_EXP`]&nbsp;&minus;&nbsp;1</sup>.
    #[unstable(feature = "f128", issue = "116909")]
    pub const MIN_POSITIVE: f128 = 3.3621031431120935062626778173217526E-4932_f128;

    /// Largest finite `f128` value.
    ///
    /// Equal to
    /// (1&nbsp;&minus;&nbsp;2<sup>&minus;[`MANTISSA_DIGITS`]</sup>)&nbsp;2<sup>[`MAX_EXP`]</sup>.
    #[unstable(feature = "f128", issue = "116909")]
    pub const MAX: f128 = 1.1897314953572317650857593266280070162e+4932_f128;

    /// One greater than the minimum possible normal power of 2 exponent.
    ///
    /// If <i>x</i>&nbsp;=&nbsp;`MIN_EXP`, then normal numbers
    /// ≥&nbsp;0.5&nbsp;×&nbsp;2<sup><i>x</i></sup>.
    #[unstable(feature = "f128", issue = "116909")]
    pub const MIN_EXP: i32 = -16381;

    /// Maximum possible power of 2 exponent.
    ///
    /// If <i>x</i>&nbsp;=&nbsp;`MAX_EXP`, then normal numbers
    /// &lt;&nbsp;1&nbsp;×&nbsp;2<sup><i>x</i></sup>.
    #[unstable(feature = "f128", issue = "116909")]
    pub const MAX_EXP: i32 = 16384;

    /// Minimum possible normal power of 10 exponent.
    ///
    /// Equal to ceil(log<sub>10</sub>&nbsp;[`MIN_POSITIVE`]).
    #[unstable(feature = "f128", issue = "116909")]
    pub const MIN_10_EXP: i32 = -4931;

    /// Maximum possible power of 10 exponent.
    ///
    /// Equal to floor(log<sub>10</sub>&nbsp;[`MAX`]).
    #[unstable(feature = "f128", issue = "116909")]
    pub const MAX_10_EXP: i32 = 4932;

    /* FIXME(f16_f128): needs const evaluation

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
    #[cfg(not(bootstrap))]
    #[rustc_diagnostic_item = "f128_nan"]
    #[unstable(feature = "f128", issue = "116909")]
    pub const NAN: f128 = 0.0_f128 / 0.0_f128;

    /// Infinity (∞).
    #[cfg(not(bootstrap))]
    #[unstable(feature = "f128", issue = "116909")]
    pub const INFINITY: f128 = 1.0_f128 / 0.0_f128;

    /// Negative infinity (−∞).
    #[cfg(not(bootstrap))]
    #[unstable(feature = "f128", issue = "116909")]
    pub const NEG_INFINITY: f128 = -1.0_f128 / 0.0_f128;

    */

    /// Raw transmutation from `u128`.
    ///
    /// This is currently identical to `transmute::<u128, f128>(v)` on all platforms.
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
    /// let v = f128::from_bits(0x4A40);
    /// assert_eq!(v, 12.5);
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "float_bits_conv", since = "1.20.0")]
    pub fn from_bits(v: u128) -> Self {
        // FIXME(f16_f128): needs const version once `classify_bits` is available
        // SAFETY: `u128` is a plain old datatype so we can always... uh...
        // ...look, just pretend you forgot what you just read.
        // Stability concerns.
        unsafe { mem::transmute::<u128, f128>(v) }
    }

    /// Raw transmutation to `u128`.
    ///
    /// This is currently identical to `transmute::<f128, u128>(self)` on all platforms.
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
    /// assert!((1f128).to_bits() != 1f128 as u128); // to_bits() is not casting!
    /// assert_eq!((12.5f128).to_bits(), 0x4A40);
    ///
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[stable(feature = "float_bits_conv", since = "1.20.0")]
    pub fn to_bits(self) -> u128 {
        // FIXME(f16_f128): needs const version once `classify_bits` is available
        // SAFETY: `u128` is a plain old datatype so we can always... uh...
        // ...look, just pretend you forgot what you just read.
        // Stability concerns.
        unsafe { mem::transmute::<f128, u128>(self) }
    }
}
