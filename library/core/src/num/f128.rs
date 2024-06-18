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

use crate::convert::FloatToInt;
use crate::mem;

/// Basic mathematical constants.
#[unstable(feature = "f128", issue = "116909")]
pub mod consts {
    // FIXME: replace with mathematical constants from cmath.

    /// Archimedes' constant (π)
    #[unstable(feature = "f128", issue = "116909")]
    pub const PI: f128 = 3.14159265358979323846264338327950288419716939937510582097494_f128;

    /// The full circle constant (τ)
    ///
    /// Equal to 2π.
    #[unstable(feature = "f128", issue = "116909")]
    pub const TAU: f128 = 6.28318530717958647692528676655900576839433879875021164194989_f128;

    /// The golden ratio (φ)
    #[unstable(feature = "f128", issue = "116909")]
    // Also, #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const PHI: f128 = 1.61803398874989484820458683436563811772030917980576286213545_f128;

    /// The Euler-Mascheroni constant (γ)
    #[unstable(feature = "f128", issue = "116909")]
    // Also, #[unstable(feature = "more_float_constants", issue = "103883")]
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
    #[unstable(feature = "f128", issue = "116909")]
    // Also, #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const FRAC_1_SQRT_PI: f128 =
        0.564189583547756286948079451560772585844050629328998856844086_f128;

    /// 1/sqrt(2π)
    #[doc(alias = "FRAC_1_SQRT_TAU")]
    #[unstable(feature = "f128", issue = "116909")]
    // Also, #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const FRAC_1_SQRT_2PI: f128 =
        0.398942280401432677939946059934381868475858631164934657665926_f128;

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
        0.707106781186547524400844362104849039284835937688474036588340_f128;

    /// sqrt(3)
    #[unstable(feature = "f128", issue = "116909")]
    // Also, #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const SQRT_3: f128 = 1.73205080756887729352744634150587236694280525381038062805581_f128;

    /// 1/sqrt(3)
    #[unstable(feature = "f128", issue = "116909")]
    // Also, #[unstable(feature = "more_float_constants", issue = "103883")]
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
    pub const LN_2: f128 = 0.693147180559945309417232121458176568075500134360255254120680_f128;

    /// ln(10)
    #[unstable(feature = "f128", issue = "116909")]
    pub const LN_10: f128 = 2.30258509299404568401799145468436420760110148862877297603333_f128;
}

#[cfg(not(test))]
impl f128 {
    // FIXME(f16_f128): almost all methods in this `impl` are missing examples and a const
    // implementation. Add these once we can run code on all platforms and have f16/f128 in CTFE.

    /// The radix or base of the internal representation of `f128`.
    #[unstable(feature = "f128", issue = "116909")]
    pub const RADIX: u32 = 2;

    /// Number of significant digits in base 2.
    #[unstable(feature = "f128", issue = "116909")]
    pub const MANTISSA_DIGITS: u32 = 113;

    /// Approximate number of significant digits in base 10.
    ///
    /// This is the maximum <i>x</i> such that any decimal number with <i>x</i>
    /// significant digits can be converted to `f128` and back without loss.
    ///
    /// Equal to floor(log<sub>10</sub>&nbsp;2<sup>[`MANTISSA_DIGITS`]&nbsp;&minus;&nbsp;1</sup>).
    ///
    /// [`MANTISSA_DIGITS`]: f128::MANTISSA_DIGITS
    #[unstable(feature = "f128", issue = "116909")]
    pub const DIGITS: u32 = 33;

    /// [Machine epsilon] value for `f128`.
    ///
    /// This is the difference between `1.0` and the next larger representable number.
    ///
    /// Equal to 2<sup>1&nbsp;&minus;&nbsp;[`MANTISSA_DIGITS`]</sup>.
    ///
    /// [Machine epsilon]: https://en.wikipedia.org/wiki/Machine_epsilon
    /// [`MANTISSA_DIGITS`]: f128::MANTISSA_DIGITS
    #[unstable(feature = "f128", issue = "116909")]
    pub const EPSILON: f128 = 1.92592994438723585305597794258492731e-34_f128;

    /// Smallest finite `f128` value.
    ///
    /// Equal to &minus;[`MAX`].
    ///
    /// [`MAX`]: f128::MAX
    #[unstable(feature = "f128", issue = "116909")]
    pub const MIN: f128 = -1.18973149535723176508575932662800701e+4932_f128;
    /// Smallest positive normal `f128` value.
    ///
    /// Equal to 2<sup>[`MIN_EXP`]&nbsp;&minus;&nbsp;1</sup>.
    ///
    /// [`MIN_EXP`]: f128::MIN_EXP
    #[unstable(feature = "f128", issue = "116909")]
    pub const MIN_POSITIVE: f128 = 3.36210314311209350626267781732175260e-4932_f128;
    /// Largest finite `f128` value.
    ///
    /// Equal to
    /// (1&nbsp;&minus;&nbsp;2<sup>&minus;[`MANTISSA_DIGITS`]</sup>)&nbsp;2<sup>[`MAX_EXP`]</sup>.
    ///
    /// [`MANTISSA_DIGITS`]: f128::MANTISSA_DIGITS
    /// [`MAX_EXP`]: f128::MAX_EXP
    #[unstable(feature = "f128", issue = "116909")]
    pub const MAX: f128 = 1.18973149535723176508575932662800701e+4932_f128;

    /// One greater than the minimum possible normal power of 2 exponent.
    ///
    /// If <i>x</i>&nbsp;=&nbsp;`MIN_EXP`, then normal numbers
    /// ≥&nbsp;0.5&nbsp;×&nbsp;2<sup><i>x</i></sup>.
    #[unstable(feature = "f128", issue = "116909")]
    pub const MIN_EXP: i32 = -16_381;
    /// Maximum possible power of 2 exponent.
    ///
    /// If <i>x</i>&nbsp;=&nbsp;`MAX_EXP`, then normal numbers
    /// &lt;&nbsp;1&nbsp;×&nbsp;2<sup><i>x</i></sup>.
    #[unstable(feature = "f128", issue = "116909")]
    pub const MAX_EXP: i32 = 16_384;

    /// Minimum <i>x</i> for which 10<sup><i>x</i></sup> is normal.
    ///
    /// Equal to ceil(log<sub>10</sub>&nbsp;[`MIN_POSITIVE`]).
    ///
    /// [`MIN_POSITIVE`]: f128::MIN_POSITIVE
    #[unstable(feature = "f128", issue = "116909")]
    pub const MIN_10_EXP: i32 = -4_931;
    /// Maximum <i>x</i> for which 10<sup><i>x</i></sup> is normal.
    ///
    /// Equal to floor(log<sub>10</sub>&nbsp;[`MAX`]).
    ///
    /// [`MAX`]: f128::MAX
    #[unstable(feature = "f128", issue = "116909")]
    pub const MAX_10_EXP: i32 = 4_932;

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
    #[allow(clippy::eq_op)]
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

    /// Sign bit
    #[cfg(not(bootstrap))]
    pub(crate) const SIGN_MASK: u128 = 0x8000_0000_0000_0000_0000_0000_0000_0000;

    /// Minimum representable positive value (min subnormal)
    #[cfg(not(bootstrap))]
    const TINY_BITS: u128 = 0x1;

    /// Minimum representable negative value (min negative subnormal)
    #[cfg(not(bootstrap))]
    const NEG_TINY_BITS: u128 = Self::TINY_BITS | Self::SIGN_MASK;

    /// Returns `true` if this value is NaN.
    ///
    /// ```
    /// #![feature(f128)]
    /// # // FIXME(f16_f128): remove when `unordtf2` is available
    /// # #[cfg(all(target_arch = "x86_64", target_os = "linux"))] {
    ///
    /// let nan = f128::NAN;
    /// let f = 7.0_f128;
    ///
    /// assert!(nan.is_nan());
    /// assert!(!f.is_nan());
    /// # }
    /// ```
    #[inline]
    #[must_use]
    #[cfg(not(bootstrap))]
    #[unstable(feature = "f128", issue = "116909")]
    #[allow(clippy::eq_op)] // > if you intended to check if the operand is NaN, use `.is_nan()` instead :)
    pub const fn is_nan(self) -> bool {
        self != self
    }

    // FIXME(#50145): `abs` is publicly unavailable in core due to
    // concerns about portability, so this implementation is for
    // private use internally.
    #[inline]
    #[cfg(not(bootstrap))]
    #[rustc_const_unstable(feature = "const_float_classify", issue = "72505")]
    pub(crate) const fn abs_private(self) -> f128 {
        // SAFETY: This transmutation is fine. Probably. For the reasons std is using it.
        unsafe {
            mem::transmute::<u128, f128>(mem::transmute::<f128, u128>(self) & !Self::SIGN_MASK)
        }
    }

    /// Returns `true` if this value is positive infinity or negative infinity, and
    /// `false` otherwise.
    ///
    /// ```
    /// #![feature(f128)]
    /// # // FIXME(f16_f128): remove when `eqtf2` is available
    /// # #[cfg(all(target_arch = "x86_64", target_os = "linux"))] {
    ///
    /// let f = 7.0f128;
    /// let inf = f128::INFINITY;
    /// let neg_inf = f128::NEG_INFINITY;
    /// let nan = f128::NAN;
    ///
    /// assert!(!f.is_infinite());
    /// assert!(!nan.is_infinite());
    ///
    /// assert!(inf.is_infinite());
    /// assert!(neg_inf.is_infinite());
    /// # }
    /// ```
    #[inline]
    #[must_use]
    #[cfg(not(bootstrap))]
    #[unstable(feature = "f128", issue = "116909")]
    #[rustc_const_unstable(feature = "const_float_classify", issue = "72505")]
    pub const fn is_infinite(self) -> bool {
        (self == f128::INFINITY) | (self == f128::NEG_INFINITY)
    }

    /// Returns `true` if this number is neither infinite nor NaN.
    ///
    /// ```
    /// #![feature(f128)]
    /// # // FIXME(f16_f128): remove when `lttf2` is available
    /// # #[cfg(all(target_arch = "x86_64", target_os = "linux"))] {
    ///
    /// let f = 7.0f128;
    /// let inf: f128 = f128::INFINITY;
    /// let neg_inf: f128 = f128::NEG_INFINITY;
    /// let nan: f128 = f128::NAN;
    ///
    /// assert!(f.is_finite());
    ///
    /// assert!(!nan.is_finite());
    /// assert!(!inf.is_finite());
    /// assert!(!neg_inf.is_finite());
    /// # }
    /// ```
    #[inline]
    #[must_use]
    #[cfg(not(bootstrap))]
    #[unstable(feature = "f128", issue = "116909")]
    #[rustc_const_unstable(feature = "const_float_classify", issue = "72505")]
    pub const fn is_finite(self) -> bool {
        // There's no need to handle NaN separately: if self is NaN,
        // the comparison is not true, exactly as desired.
        self.abs_private() < Self::INFINITY
    }

    /// Returns `true` if `self` has a positive sign, including `+0.0`, NaNs with
    /// positive sign bit and positive infinity. Note that IEEE 754 doesn't assign any
    /// meaning to the sign bit in case of a NaN, and as Rust doesn't guarantee that
    /// the bit pattern of NaNs are conserved over arithmetic operations, the result of
    /// `is_sign_positive` on a NaN might produce an unexpected result in some cases.
    /// See [explanation of NaN as a special value](f128) for more info.
    ///
    /// ```
    /// #![feature(f128)]
    ///
    /// let f = 7.0_f128;
    /// let g = -7.0_f128;
    ///
    /// assert!(f.is_sign_positive());
    /// assert!(!g.is_sign_positive());
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "f128", issue = "116909")]
    pub fn is_sign_positive(self) -> bool {
        !self.is_sign_negative()
    }

    /// Returns `true` if `self` has a negative sign, including `-0.0`, NaNs with
    /// negative sign bit and negative infinity. Note that IEEE 754 doesn't assign any
    /// meaning to the sign bit in case of a NaN, and as Rust doesn't guarantee that
    /// the bit pattern of NaNs are conserved over arithmetic operations, the result of
    /// `is_sign_negative` on a NaN might produce an unexpected result in some cases.
    /// See [explanation of NaN as a special value](f128) for more info.
    ///
    /// ```
    /// #![feature(f128)]
    ///
    /// let f = 7.0_f128;
    /// let g = -7.0_f128;
    ///
    /// assert!(!f.is_sign_negative());
    /// assert!(g.is_sign_negative());
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "f128", issue = "116909")]
    pub fn is_sign_negative(self) -> bool {
        // IEEE754 says: isSignMinus(x) is true if and only if x has negative sign. isSignMinus
        // applies to zeros and NaNs as well.
        // SAFETY: This is just transmuting to get the sign bit, it's fine.
        (self.to_bits() & (1 << 127)) != 0
    }

    /// Returns the least number greater than `self`.
    ///
    /// Let `TINY` be the smallest representable positive `f128`. Then,
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
    /// #![feature(f128)]
    /// #![feature(float_next_up_down)]
    /// # // FIXME(f16_f128): remove when `eqtf2` is available
    /// # #[cfg(all(target_arch = "x86_64", target_os = "linux"))] {
    ///
    /// // f128::EPSILON is the difference between 1.0 and the next number up.
    /// assert_eq!(1.0f128.next_up(), 1.0 + f128::EPSILON);
    /// // But not for most numbers.
    /// assert!(0.1f128.next_up() < 0.1 + f128::EPSILON);
    /// assert_eq!(4611686018427387904f128.next_up(), 4611686018427387904.000000000000001);
    /// # }
    /// ```
    ///
    /// [`NEG_INFINITY`]: Self::NEG_INFINITY
    /// [`INFINITY`]: Self::INFINITY
    /// [`MIN`]: Self::MIN
    /// [`MAX`]: Self::MAX
    #[inline]
    #[cfg(not(bootstrap))]
    #[unstable(feature = "f128", issue = "116909")]
    // #[unstable(feature = "float_next_up_down", issue = "91399")]
    pub fn next_up(self) -> Self {
        // Some targets violate Rust's assumption of IEEE semantics, e.g. by flushing
        // denormals to zero. This is in general unsound and unsupported, but here
        // we do our best to still produce the correct result on such targets.
        let bits = self.to_bits();
        if self.is_nan() || bits == Self::INFINITY.to_bits() {
            return self;
        }

        let abs = bits & !Self::SIGN_MASK;
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
    /// Let `TINY` be the smallest representable positive `f128`. Then,
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
    /// #![feature(f128)]
    /// #![feature(float_next_up_down)]
    /// # // FIXME(f16_f128): remove when `eqtf2` is available
    /// # #[cfg(all(target_arch = "x86_64", target_os = "linux"))] {
    ///
    /// let x = 1.0f128;
    /// // Clamp value into range [0, 1).
    /// let clamped = x.clamp(0.0, 1.0f128.next_down());
    /// assert!(clamped < 1.0);
    /// assert_eq!(clamped.next_up(), 1.0);
    /// # }
    /// ```
    ///
    /// [`NEG_INFINITY`]: Self::NEG_INFINITY
    /// [`INFINITY`]: Self::INFINITY
    /// [`MIN`]: Self::MIN
    /// [`MAX`]: Self::MAX
    #[inline]
    #[cfg(not(bootstrap))]
    #[unstable(feature = "f128", issue = "116909")]
    // #[unstable(feature = "float_next_up_down", issue = "91399")]
    pub fn next_down(self) -> Self {
        // Some targets violate Rust's assumption of IEEE semantics, e.g. by flushing
        // denormals to zero. This is in general unsound and unsupported, but here
        // we do our best to still produce the correct result on such targets.
        let bits = self.to_bits();
        if self.is_nan() || bits == Self::NEG_INFINITY.to_bits() {
            return self;
        }

        let abs = bits & !Self::SIGN_MASK;
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
    /// #![feature(f128)]
    /// # // FIXME(f16_f128): remove when `eqtf2` is available
    /// # #[cfg(all(target_arch = "x86_64", target_os = "linux"))] {
    ///
    /// let x = 2.0_f128;
    /// let abs_difference = (x.recip() - (1.0 / x)).abs();
    ///
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[cfg(not(bootstrap))]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub fn recip(self) -> Self {
        1.0 / self
    }

    /// Converts radians to degrees.
    ///
    /// ```
    /// #![feature(f128)]
    /// # // FIXME(f16_f128): remove when `eqtf2` is available
    /// # #[cfg(all(target_arch = "x86_64", target_os = "linux"))] {
    ///
    /// let angle = std::f128::consts::PI;
    ///
    /// let abs_difference = (angle.to_degrees() - 180.0).abs();
    /// assert!(abs_difference <= f128::EPSILON);
    /// # }
    /// ```
    #[inline]
    #[cfg(not(bootstrap))]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub fn to_degrees(self) -> Self {
        // Use a literal for better precision.
        const PIS_IN_180: f128 = 57.2957795130823208767981548141051703324054724665643215491602_f128;
        self * PIS_IN_180
    }

    /// Converts degrees to radians.
    ///
    /// ```
    /// #![feature(f128)]
    /// # // FIXME(f16_f128): remove when `eqtf2` is available
    /// # #[cfg(all(target_arch = "x86_64", target_os = "linux"))] {
    ///
    /// let angle = 180.0f128;
    ///
    /// let abs_difference = (angle.to_radians() - std::f128::consts::PI).abs();
    ///
    /// assert!(abs_difference <= 1e-30);
    /// # }
    /// ```
    #[inline]
    #[cfg(not(bootstrap))]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub fn to_radians(self) -> f128 {
        // Use a literal for better precision.
        const RADS_PER_DEG: f128 =
            0.0174532925199432957692369076848861271344287188854172545609719_f128;
        self * RADS_PER_DEG
    }

    /// Rounds toward zero and converts to any primitive integer type,
    /// assuming that the value is finite and fits in that type.
    ///
    /// ```
    /// #![feature(f128)]
    /// # // FIXME(f16_f128): remove when `float*itf` is available
    /// # #[cfg(all(target_arch = "x86_64", target_os = "linux"))] {
    ///
    /// let value = 4.6_f128;
    /// let rounded = unsafe { value.to_int_unchecked::<u16>() };
    /// assert_eq!(rounded, 4);
    ///
    /// let value = -128.9_f128;
    /// let rounded = unsafe { value.to_int_unchecked::<i8>() };
    /// assert_eq!(rounded, i8::MIN);
    /// # }
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
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub unsafe fn to_int_unchecked<Int>(self) -> Int
    where
        Self: FloatToInt<Int>,
    {
        // SAFETY: the caller must uphold the safety contract for
        // `FloatToInt::to_int_unchecked`.
        unsafe { FloatToInt::<Int>::to_int_unchecked(self) }
    }

    /// Raw transmutation to `u128`.
    ///
    /// This is currently identical to `transmute::<f128, u128>(self)` on all platforms.
    ///
    /// See [`from_bits`](#method.from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// Note that this function is distinct from `as` casting, which attempts to
    /// preserve the *numeric* value, and not the bitwise value.
    ///
    /// ```
    /// #![feature(f128)]
    ///
    /// # // FIXME(f16_f128): enable this once const casting works
    /// # // assert_ne!((1f128).to_bits(), 1f128 as u128); // to_bits() is not casting!
    /// assert_eq!((12.5f128).to_bits(), 0x40029000000000000000000000000000);
    /// ```
    #[inline]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub fn to_bits(self) -> u128 {
        // SAFETY: `u128` is a plain old datatype so we can always... uh...
        // ...look, just pretend you forgot what you just read.
        // Stability concerns.
        unsafe { mem::transmute(self) }
    }

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
    /// If you don't care about signalingness (very likely), then there is no
    /// portability concern.
    ///
    /// Note that this function is distinct from `as` casting, which attempts to
    /// preserve the *numeric* value, and not the bitwise value.
    ///
    /// ```
    /// #![feature(f128)]
    /// #  // FIXME(f16_f128): remove when `eqtf2` is available
    /// # #[cfg(all(target_arch = "x86_64", target_os = "linux"))] {
    ///
    /// let v = f128::from_bits(0x40029000000000000000000000000000);
    /// assert_eq!(v, 12.5);
    /// # }
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "f128", issue = "116909")]
    pub fn from_bits(v: u128) -> Self {
        // SAFETY: `u128 is a plain old datatype so we can always... uh...
        // ...look, just pretend you forgot what you just read.
        // Stability concerns.
        unsafe { mem::transmute(v) }
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
    /// #![feature(f128)]
    ///
    /// let bytes = 12.5f128.to_be_bytes();
    /// assert_eq!(
    ///     bytes,
    ///     [0x40, 0x02, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00,
    ///      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    /// );
    /// ```
    #[inline]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub fn to_be_bytes(self) -> [u8; 16] {
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
    /// #![feature(f128)]
    ///
    /// let bytes = 12.5f128.to_le_bytes();
    /// assert_eq!(
    ///     bytes,
    ///     [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ///      0x00, 0x00, 0x00, 0x00, 0x00, 0x90, 0x02, 0x40]
    /// );
    /// ```
    #[inline]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub fn to_le_bytes(self) -> [u8; 16] {
        self.to_bits().to_le_bytes()
    }

    /// Return the memory representation of this floating point number as a byte array in
    /// native byte order.
    ///
    /// As the target platform's native endianness is used, portable code
    /// should use [`to_be_bytes`] or [`to_le_bytes`], as appropriate, instead.
    ///
    /// [`to_be_bytes`]: f128::to_be_bytes
    /// [`to_le_bytes`]: f128::to_le_bytes
    ///
    /// See [`from_bits`](Self::from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    ///
    /// let bytes = 12.5f128.to_ne_bytes();
    /// assert_eq!(
    ///     bytes,
    ///     if cfg!(target_endian = "big") {
    ///         [0x40, 0x02, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00,
    ///          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    ///     } else {
    ///         [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ///          0x00, 0x00, 0x00, 0x00, 0x00, 0x90, 0x02, 0x40]
    ///     }
    /// );
    /// ```
    #[inline]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub fn to_ne_bytes(self) -> [u8; 16] {
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
    /// #![feature(f128)]
    /// # // FIXME(f16_f128): remove when `eqtf2` is available
    /// # #[cfg(all(target_arch = "x86_64", target_os = "linux"))] {
    ///
    /// let value = f128::from_be_bytes(
    ///     [0x40, 0x02, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00,
    ///      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    /// );
    /// assert_eq!(value, 12.5);
    /// # }
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "f128", issue = "116909")]
    pub fn from_be_bytes(bytes: [u8; 16]) -> Self {
        Self::from_bits(u128::from_be_bytes(bytes))
    }

    /// Create a floating point value from its representation as a byte array in little endian.
    ///
    /// See [`from_bits`](Self::from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # // FIXME(f16_f128): remove when `eqtf2` is available
    /// # #[cfg(all(target_arch = "x86_64", target_os = "linux"))] {
    ///
    /// let value = f128::from_le_bytes(
    ///     [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ///      0x00, 0x00, 0x00, 0x00, 0x00, 0x90, 0x02, 0x40]
    /// );
    /// assert_eq!(value, 12.5);
    /// # }
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "f128", issue = "116909")]
    pub fn from_le_bytes(bytes: [u8; 16]) -> Self {
        Self::from_bits(u128::from_le_bytes(bytes))
    }

    /// Create a floating point value from its representation as a byte array in native endian.
    ///
    /// As the target platform's native endianness is used, portable code
    /// likely wants to use [`from_be_bytes`] or [`from_le_bytes`], as
    /// appropriate instead.
    ///
    /// [`from_be_bytes`]: f128::from_be_bytes
    /// [`from_le_bytes`]: f128::from_le_bytes
    ///
    /// See [`from_bits`](Self::from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(f128)]
    /// # // FIXME(f16_f128): remove when `eqtf2` is available
    /// # #[cfg(all(target_arch = "x86_64", target_os = "linux"))] {
    ///
    /// let value = f128::from_ne_bytes(if cfg!(target_endian = "big") {
    ///     [0x40, 0x02, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00,
    ///      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    /// } else {
    ///     [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ///      0x00, 0x00, 0x00, 0x00, 0x00, 0x90, 0x02, 0x40]
    /// });
    /// assert_eq!(value, 12.5);
    /// # }
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "f128", issue = "116909")]
    pub fn from_ne_bytes(bytes: [u8; 16]) -> Self {
        Self::from_bits(u128::from_ne_bytes(bytes))
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
    /// [`PartialOrd`] and [`PartialEq`] implementations of `f128`. For example,
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
    /// #![feature(f128)]
    ///
    /// struct GoodBoy {
    ///     name: &'static str,
    ///     weight: f128,
    /// }
    ///
    /// let mut bois = vec![
    ///     GoodBoy { name: "Pucci", weight: 0.1 },
    ///     GoodBoy { name: "Woofer", weight: 99.0 },
    ///     GoodBoy { name: "Yapper", weight: 10.0 },
    ///     GoodBoy { name: "Chonk", weight: f128::INFINITY },
    ///     GoodBoy { name: "Abs. Unit", weight: f128::NAN },
    ///     GoodBoy { name: "Floaty", weight: -5.0 },
    /// ];
    ///
    /// bois.sort_by(|a, b| a.weight.total_cmp(&b.weight));
    ///
    /// // `f128::NAN` could be positive or negative, which will affect the sort order.
    /// if f128::NAN.is_sign_negative() {
    ///     bois.into_iter().map(|b| b.weight)
    ///         .zip([f128::NAN, -5.0, 0.1, 10.0, 99.0, f128::INFINITY].iter())
    ///         .for_each(|(a, b)| assert_eq!(a.to_bits(), b.to_bits()))
    /// } else {
    ///     bois.into_iter().map(|b| b.weight)
    ///         .zip([-5.0, 0.1, 10.0, 99.0, f128::INFINITY, f128::NAN].iter())
    ///         .for_each(|(a, b)| assert_eq!(a.to_bits(), b.to_bits()))
    /// }
    /// ```
    #[inline]
    #[must_use]
    #[cfg(not(bootstrap))]
    #[unstable(feature = "f128", issue = "116909")]
    pub fn total_cmp(&self, other: &Self) -> crate::cmp::Ordering {
        let mut left = self.to_bits() as i128;
        let mut right = other.to_bits() as i128;

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
        left ^= (((left >> 127) as u128) >> 1) as i128;
        right ^= (((right >> 127) as u128) >> 1) as i128;

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
    /// #![feature(f128)]
    /// # // FIXME(f16_f128): remove when `{eq,gt,unord}tf` are available
    /// # #[cfg(all(target_arch = "x86_64", target_os = "linux"))] {
    ///
    /// assert!((-3.0f128).clamp(-2.0, 1.0) == -2.0);
    /// assert!((0.0f128).clamp(-2.0, 1.0) == 0.0);
    /// assert!((2.0f128).clamp(-2.0, 1.0) == 1.0);
    /// assert!((f128::NAN).clamp(-2.0, 1.0).is_nan());
    /// # }
    /// ```
    #[inline]
    #[cfg(not(bootstrap))]
    #[unstable(feature = "f128", issue = "116909")]
    #[must_use = "method returns a new number and does not mutate the original value"]
    pub fn clamp(mut self, min: f128, max: f128) -> f128 {
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
