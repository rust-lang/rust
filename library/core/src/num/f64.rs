//! Constants for the `f64` double-precision floating point type.
//!
//! *[See also the `f64` primitive type][f64].*
//!
//! Mathematically significant numbers are provided in the `consts` sub-module.
//!
//! For the constants defined directly in this module
//! (as distinct from those defined in the `consts` sub-module),
//! new code should instead use the associated constants
//! defined directly on the `f64` type.

#![stable(feature = "rust1", since = "1.0.0")]

use crate::convert::FloatToInt;
#[cfg(not(test))]
use crate::intrinsics;
use crate::mem;
use crate::num::FpCategory;
use crate::panic::const_assert;

/// The radix or base of the internal representation of `f64`.
/// Use [`f64::RADIX`] instead.
///
/// # Examples
///
/// ```rust
/// // deprecated way
/// # #[allow(deprecated, deprecated_in_future)]
/// let r = std::f64::RADIX;
///
/// // intended way
/// let r = f64::RADIX;
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(since = "TBD", note = "replaced by the `RADIX` associated constant on `f64`")]
#[rustc_diagnostic_item = "f64_legacy_const_radix"]
pub const RADIX: u32 = f64::RADIX;

/// Number of significant digits in base 2.
/// Use [`f64::MANTISSA_DIGITS`] instead.
///
/// # Examples
///
/// ```rust
/// // deprecated way
/// # #[allow(deprecated, deprecated_in_future)]
/// let d = std::f64::MANTISSA_DIGITS;
///
/// // intended way
/// let d = f64::MANTISSA_DIGITS;
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(
    since = "TBD",
    note = "replaced by the `MANTISSA_DIGITS` associated constant on `f64`"
)]
#[rustc_diagnostic_item = "f64_legacy_const_mantissa_dig"]
pub const MANTISSA_DIGITS: u32 = f64::MANTISSA_DIGITS;

/// Approximate number of significant digits in base 10.
/// Use [`f64::DIGITS`] instead.
///
/// # Examples
///
/// ```rust
/// // deprecated way
/// # #[allow(deprecated, deprecated_in_future)]
/// let d = std::f64::DIGITS;
///
/// // intended way
/// let d = f64::DIGITS;
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(since = "TBD", note = "replaced by the `DIGITS` associated constant on `f64`")]
#[rustc_diagnostic_item = "f64_legacy_const_digits"]
pub const DIGITS: u32 = f64::DIGITS;

/// [Machine epsilon] value for `f64`.
/// Use [`f64::EPSILON`] instead.
///
/// This is the difference between `1.0` and the next larger representable number.
///
/// [Machine epsilon]: https://en.wikipedia.org/wiki/Machine_epsilon
///
/// # Examples
///
/// ```rust
/// // deprecated way
/// # #[allow(deprecated, deprecated_in_future)]
/// let e = std::f64::EPSILON;
///
/// // intended way
/// let e = f64::EPSILON;
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(since = "TBD", note = "replaced by the `EPSILON` associated constant on `f64`")]
#[rustc_diagnostic_item = "f64_legacy_const_epsilon"]
pub const EPSILON: f64 = f64::EPSILON;

/// Smallest finite `f64` value.
/// Use [`f64::MIN`] instead.
///
/// # Examples
///
/// ```rust
/// // deprecated way
/// # #[allow(deprecated, deprecated_in_future)]
/// let min = std::f64::MIN;
///
/// // intended way
/// let min = f64::MIN;
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(since = "TBD", note = "replaced by the `MIN` associated constant on `f64`")]
#[rustc_diagnostic_item = "f64_legacy_const_min"]
pub const MIN: f64 = f64::MIN;

/// Smallest positive normal `f64` value.
/// Use [`f64::MIN_POSITIVE`] instead.
///
/// # Examples
///
/// ```rust
/// // deprecated way
/// # #[allow(deprecated, deprecated_in_future)]
/// let min = std::f64::MIN_POSITIVE;
///
/// // intended way
/// let min = f64::MIN_POSITIVE;
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(since = "TBD", note = "replaced by the `MIN_POSITIVE` associated constant on `f64`")]
#[rustc_diagnostic_item = "f64_legacy_const_min_positive"]
pub const MIN_POSITIVE: f64 = f64::MIN_POSITIVE;

/// Largest finite `f64` value.
/// Use [`f64::MAX`] instead.
///
/// # Examples
///
/// ```rust
/// // deprecated way
/// # #[allow(deprecated, deprecated_in_future)]
/// let max = std::f64::MAX;
///
/// // intended way
/// let max = f64::MAX;
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(since = "TBD", note = "replaced by the `MAX` associated constant on `f64`")]
#[rustc_diagnostic_item = "f64_legacy_const_max"]
pub const MAX: f64 = f64::MAX;

/// One greater than the minimum possible normal power of 2 exponent.
/// Use [`f64::MIN_EXP`] instead.
///
/// # Examples
///
/// ```rust
/// // deprecated way
/// # #[allow(deprecated, deprecated_in_future)]
/// let min = std::f64::MIN_EXP;
///
/// // intended way
/// let min = f64::MIN_EXP;
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(since = "TBD", note = "replaced by the `MIN_EXP` associated constant on `f64`")]
#[rustc_diagnostic_item = "f64_legacy_const_min_exp"]
pub const MIN_EXP: i32 = f64::MIN_EXP;

/// Maximum possible power of 2 exponent.
/// Use [`f64::MAX_EXP`] instead.
///
/// # Examples
///
/// ```rust
/// // deprecated way
/// # #[allow(deprecated, deprecated_in_future)]
/// let max = std::f64::MAX_EXP;
///
/// // intended way
/// let max = f64::MAX_EXP;
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(since = "TBD", note = "replaced by the `MAX_EXP` associated constant on `f64`")]
#[rustc_diagnostic_item = "f64_legacy_const_max_exp"]
pub const MAX_EXP: i32 = f64::MAX_EXP;

/// Minimum possible normal power of 10 exponent.
/// Use [`f64::MIN_10_EXP`] instead.
///
/// # Examples
///
/// ```rust
/// // deprecated way
/// # #[allow(deprecated, deprecated_in_future)]
/// let min = std::f64::MIN_10_EXP;
///
/// // intended way
/// let min = f64::MIN_10_EXP;
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(since = "TBD", note = "replaced by the `MIN_10_EXP` associated constant on `f64`")]
#[rustc_diagnostic_item = "f64_legacy_const_min_10_exp"]
pub const MIN_10_EXP: i32 = f64::MIN_10_EXP;

/// Maximum possible power of 10 exponent.
/// Use [`f64::MAX_10_EXP`] instead.
///
/// # Examples
///
/// ```rust
/// // deprecated way
/// # #[allow(deprecated, deprecated_in_future)]
/// let max = std::f64::MAX_10_EXP;
///
/// // intended way
/// let max = f64::MAX_10_EXP;
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(since = "TBD", note = "replaced by the `MAX_10_EXP` associated constant on `f64`")]
#[rustc_diagnostic_item = "f64_legacy_const_max_10_exp"]
pub const MAX_10_EXP: i32 = f64::MAX_10_EXP;

/// Not a Number (NaN).
/// Use [`f64::NAN`] instead.
///
/// # Examples
///
/// ```rust
/// // deprecated way
/// # #[allow(deprecated, deprecated_in_future)]
/// let nan = std::f64::NAN;
///
/// // intended way
/// let nan = f64::NAN;
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(since = "TBD", note = "replaced by the `NAN` associated constant on `f64`")]
#[rustc_diagnostic_item = "f64_legacy_const_nan"]
pub const NAN: f64 = f64::NAN;

/// Infinity (∞).
/// Use [`f64::INFINITY`] instead.
///
/// # Examples
///
/// ```rust
/// // deprecated way
/// # #[allow(deprecated, deprecated_in_future)]
/// let inf = std::f64::INFINITY;
///
/// // intended way
/// let inf = f64::INFINITY;
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(since = "TBD", note = "replaced by the `INFINITY` associated constant on `f64`")]
#[rustc_diagnostic_item = "f64_legacy_const_infinity"]
pub const INFINITY: f64 = f64::INFINITY;

/// Negative infinity (−∞).
/// Use [`f64::NEG_INFINITY`] instead.
///
/// # Examples
///
/// ```rust
/// // deprecated way
/// # #[allow(deprecated, deprecated_in_future)]
/// let ninf = std::f64::NEG_INFINITY;
///
/// // intended way
/// let ninf = f64::NEG_INFINITY;
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(since = "TBD", note = "replaced by the `NEG_INFINITY` associated constant on `f64`")]
#[rustc_diagnostic_item = "f64_legacy_const_neg_infinity"]
pub const NEG_INFINITY: f64 = f64::NEG_INFINITY;

/// Basic mathematical constants.
#[stable(feature = "rust1", since = "1.0.0")]
pub mod consts {
    // FIXME: replace with mathematical constants from cmath.

    /// Archimedes' constant (π)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const PI: f64 = 3.14159265358979323846264338327950288_f64;

    /// The full circle constant (τ)
    ///
    /// Equal to 2π.
    #[stable(feature = "tau_constant", since = "1.47.0")]
    pub const TAU: f64 = 6.28318530717958647692528676655900577_f64;

    /// The golden ratio (φ)
    #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const PHI: f64 = 1.618033988749894848204586834365638118_f64;

    /// The Euler-Mascheroni constant (γ)
    #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const EGAMMA: f64 = 0.577215664901532860606512090082402431_f64;

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

    /// 1/sqrt(π)
    #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const FRAC_1_SQRT_PI: f64 = 0.564189583547756286948079451560772586_f64;

    /// 1/sqrt(2π)
    #[doc(alias = "FRAC_1_SQRT_TAU")]
    #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const FRAC_1_SQRT_2PI: f64 = 0.398942280401432677939946059934381868_f64;

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

    /// sqrt(3)
    #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const SQRT_3: f64 = 1.732050807568877293527446341505872367_f64;

    /// 1/sqrt(3)
    #[unstable(feature = "more_float_constants", issue = "103883")]
    pub const FRAC_1_SQRT_3: f64 = 0.577350269189625764509148780501957456_f64;

    /// Euler's number (e)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const E: f64 = 2.71828182845904523536028747135266250_f64;

    /// log<sub>2</sub>(10)
    #[stable(feature = "extra_log_consts", since = "1.43.0")]
    pub const LOG2_10: f64 = 3.32192809488736234787031942948939018_f64;

    /// log<sub>2</sub>(e)
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const LOG2_E: f64 = 1.44269504088896340735992468100189214_f64;

    /// log<sub>10</sub>(2)
    #[stable(feature = "extra_log_consts", since = "1.43.0")]
    pub const LOG10_2: f64 = 0.301029995663981195213738894724493027_f64;

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

#[cfg(not(test))]
impl f64 {
    /// The radix or base of the internal representation of `f64`.
    #[stable(feature = "assoc_int_consts", since = "1.43.0")]
    pub const RADIX: u32 = 2;

    /// Number of significant digits in base 2.
    #[stable(feature = "assoc_int_consts", since = "1.43.0")]
    pub const MANTISSA_DIGITS: u32 = 53;
    /// Approximate number of significant digits in base 10.
    ///
    /// This is the maximum <i>x</i> such that any decimal number with <i>x</i>
    /// significant digits can be converted to `f64` and back without loss.
    ///
    /// Equal to floor(log<sub>10</sub>&nbsp;2<sup>[`MANTISSA_DIGITS`]&nbsp;&minus;&nbsp;1</sup>).
    ///
    /// [`MANTISSA_DIGITS`]: f64::MANTISSA_DIGITS
    #[stable(feature = "assoc_int_consts", since = "1.43.0")]
    pub const DIGITS: u32 = 15;

    /// [Machine epsilon] value for `f64`.
    ///
    /// This is the difference between `1.0` and the next larger representable number.
    ///
    /// Equal to 2<sup>1&nbsp;&minus;&nbsp;[`MANTISSA_DIGITS`]</sup>.
    ///
    /// [Machine epsilon]: https://en.wikipedia.org/wiki/Machine_epsilon
    /// [`MANTISSA_DIGITS`]: f64::MANTISSA_DIGITS
    #[stable(feature = "assoc_int_consts", since = "1.43.0")]
    #[cfg_attr(not(test), rustc_diagnostic_item = "f64_epsilon")]
    pub const EPSILON: f64 = 2.2204460492503131e-16_f64;

    /// Smallest finite `f64` value.
    ///
    /// Equal to &minus;[`MAX`].
    ///
    /// [`MAX`]: f64::MAX
    #[stable(feature = "assoc_int_consts", since = "1.43.0")]
    pub const MIN: f64 = -1.7976931348623157e+308_f64;
    /// Smallest positive normal `f64` value.
    ///
    /// Equal to 2<sup>[`MIN_EXP`]&nbsp;&minus;&nbsp;1</sup>.
    ///
    /// [`MIN_EXP`]: f64::MIN_EXP
    #[stable(feature = "assoc_int_consts", since = "1.43.0")]
    pub const MIN_POSITIVE: f64 = 2.2250738585072014e-308_f64;
    /// Largest finite `f64` value.
    ///
    /// Equal to
    /// (1&nbsp;&minus;&nbsp;2<sup>&minus;[`MANTISSA_DIGITS`]</sup>)&nbsp;2<sup>[`MAX_EXP`]</sup>.
    ///
    /// [`MANTISSA_DIGITS`]: f64::MANTISSA_DIGITS
    /// [`MAX_EXP`]: f64::MAX_EXP
    #[stable(feature = "assoc_int_consts", since = "1.43.0")]
    pub const MAX: f64 = 1.7976931348623157e+308_f64;

    /// One greater than the minimum possible normal power of 2 exponent.
    ///
    /// If <i>x</i>&nbsp;=&nbsp;`MIN_EXP`, then normal numbers
    /// ≥&nbsp;0.5&nbsp;×&nbsp;2<sup><i>x</i></sup>.
    #[stable(feature = "assoc_int_consts", since = "1.43.0")]
    pub const MIN_EXP: i32 = -1021;
    /// Maximum possible power of 2 exponent.
    ///
    /// If <i>x</i>&nbsp;=&nbsp;`MAX_EXP`, then normal numbers
    /// &lt;&nbsp;1&nbsp;×&nbsp;2<sup><i>x</i></sup>.
    #[stable(feature = "assoc_int_consts", since = "1.43.0")]
    pub const MAX_EXP: i32 = 1024;

    /// Minimum <i>x</i> for which 10<sup><i>x</i></sup> is normal.
    ///
    /// Equal to ceil(log<sub>10</sub>&nbsp;[`MIN_POSITIVE`]).
    ///
    /// [`MIN_POSITIVE`]: f64::MIN_POSITIVE
    #[stable(feature = "assoc_int_consts", since = "1.43.0")]
    pub const MIN_10_EXP: i32 = -307;
    /// Maximum <i>x</i> for which 10<sup><i>x</i></sup> is normal.
    ///
    /// Equal to floor(log<sub>10</sub>&nbsp;[`MAX`]).
    ///
    /// [`MAX`]: f64::MAX
    #[stable(feature = "assoc_int_consts", since = "1.43.0")]
    pub const MAX_10_EXP: i32 = 308;

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
    #[rustc_diagnostic_item = "f64_nan"]
    #[stable(feature = "assoc_int_consts", since = "1.43.0")]
    #[allow(clippy::eq_op)]
    pub const NAN: f64 = 0.0_f64 / 0.0_f64;
    /// Infinity (∞).
    #[stable(feature = "assoc_int_consts", since = "1.43.0")]
    pub const INFINITY: f64 = 1.0_f64 / 0.0_f64;
    /// Negative infinity (−∞).
    #[stable(feature = "assoc_int_consts", since = "1.43.0")]
    pub const NEG_INFINITY: f64 = -1.0_f64 / 0.0_f64;

    /// Sign bit
    const SIGN_MASK: u64 = 0x8000_0000_0000_0000;

    /// Exponent mask
    const EXP_MASK: u64 = 0x7ff0_0000_0000_0000;

    /// Mantissa mask
    const MAN_MASK: u64 = 0x000f_ffff_ffff_ffff;

    /// Minimum representable positive value (min subnormal)
    const TINY_BITS: u64 = 0x1;

    /// Minimum representable negative value (min negative subnormal)
    const NEG_TINY_BITS: u64 = Self::TINY_BITS | Self::SIGN_MASK;

    /// Returns `true` if this value is NaN.
    ///
    /// ```
    /// let nan = f64::NAN;
    /// let f = 7.0_f64;
    ///
    /// assert!(nan.is_nan());
    /// assert!(!f.is_nan());
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_classify", since = "1.83.0")]
    #[inline]
    #[allow(clippy::eq_op)] // > if you intended to check if the operand is NaN, use `.is_nan()` instead :)
    pub const fn is_nan(self) -> bool {
        self != self
    }

    /// Returns `true` if this value is positive infinity or negative infinity, and
    /// `false` otherwise.
    ///
    /// ```
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_classify", since = "1.83.0")]
    #[inline]
    pub const fn is_infinite(self) -> bool {
        // Getting clever with transmutation can result in incorrect answers on some FPUs
        // FIXME: alter the Rust <-> Rust calling convention to prevent this problem.
        // See https://github.com/rust-lang/rust/issues/72327
        (self == f64::INFINITY) | (self == f64::NEG_INFINITY)
    }

    /// Returns `true` if this number is neither infinite nor NaN.
    ///
    /// ```
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_classify", since = "1.83.0")]
    #[inline]
    pub const fn is_finite(self) -> bool {
        // There's no need to handle NaN separately: if self is NaN,
        // the comparison is not true, exactly as desired.
        self.abs() < Self::INFINITY
    }

    /// Returns `true` if the number is [subnormal].
    ///
    /// ```
    /// let min = f64::MIN_POSITIVE; // 2.2250738585072014e-308_f64
    /// let max = f64::MAX;
    /// let lower_than_min = 1.0e-308_f64;
    /// let zero = 0.0_f64;
    ///
    /// assert!(!min.is_subnormal());
    /// assert!(!max.is_subnormal());
    ///
    /// assert!(!zero.is_subnormal());
    /// assert!(!f64::NAN.is_subnormal());
    /// assert!(!f64::INFINITY.is_subnormal());
    /// // Values between `0` and `min` are Subnormal.
    /// assert!(lower_than_min.is_subnormal());
    /// ```
    /// [subnormal]: https://en.wikipedia.org/wiki/Denormal_number
    #[must_use]
    #[stable(feature = "is_subnormal", since = "1.53.0")]
    #[rustc_const_stable(feature = "const_float_classify", since = "1.83.0")]
    #[inline]
    pub const fn is_subnormal(self) -> bool {
        matches!(self.classify(), FpCategory::Subnormal)
    }

    /// Returns `true` if the number is neither zero, infinite,
    /// [subnormal], or NaN.
    ///
    /// ```
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_classify", since = "1.83.0")]
    #[inline]
    pub const fn is_normal(self) -> bool {
        matches!(self.classify(), FpCategory::Normal)
    }

    /// Returns the floating point category of the number. If only one property
    /// is going to be tested, it is generally faster to use the specific
    /// predicate instead.
    ///
    /// ```
    /// use std::num::FpCategory;
    ///
    /// let num = 12.4_f64;
    /// let inf = f64::INFINITY;
    ///
    /// assert_eq!(num.classify(), FpCategory::Normal);
    /// assert_eq!(inf.classify(), FpCategory::Infinite);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_classify", since = "1.83.0")]
    pub const fn classify(self) -> FpCategory {
        // We used to have complicated logic here that avoids the simple bit-based tests to work
        // around buggy codegen for x87 targets (see
        // https://github.com/rust-lang/rust/issues/114479). However, some LLVM versions later, none
        // of our tests is able to find any difference between the complicated and the naive
        // version, so now we are back to the naive version.
        let b = self.to_bits();
        match (b & Self::MAN_MASK, b & Self::EXP_MASK) {
            (0, Self::EXP_MASK) => FpCategory::Infinite,
            (_, Self::EXP_MASK) => FpCategory::Nan,
            (0, 0) => FpCategory::Zero,
            (_, 0) => FpCategory::Subnormal,
            _ => FpCategory::Normal,
        }
    }

    /// Returns `true` if `self` has a positive sign, including `+0.0`, NaNs with
    /// positive sign bit and positive infinity.
    ///
    /// Note that IEEE 754 doesn't assign any meaning to the sign bit in case of
    /// a NaN, and as Rust doesn't guarantee that the bit pattern of NaNs are
    /// conserved over arithmetic operations, the result of `is_sign_positive` on
    /// a NaN might produce an unexpected or non-portable result. See the [specification
    /// of NaN bit patterns](f32#nan-bit-patterns) for more info. Use `self.signum() == 1.0`
    /// if you need fully portable behavior (will return `false` for all NaNs).
    ///
    /// ```
    /// let f = 7.0_f64;
    /// let g = -7.0_f64;
    ///
    /// assert!(f.is_sign_positive());
    /// assert!(!g.is_sign_positive());
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_classify", since = "1.83.0")]
    #[inline]
    pub const fn is_sign_positive(self) -> bool {
        !self.is_sign_negative()
    }

    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[deprecated(since = "1.0.0", note = "renamed to is_sign_positive")]
    #[inline]
    #[doc(hidden)]
    pub fn is_positive(self) -> bool {
        self.is_sign_positive()
    }

    /// Returns `true` if `self` has a negative sign, including `-0.0`, NaNs with
    /// negative sign bit and negative infinity.
    ///
    /// Note that IEEE 754 doesn't assign any meaning to the sign bit in case of
    /// a NaN, and as Rust doesn't guarantee that the bit pattern of NaNs are
    /// conserved over arithmetic operations, the result of `is_sign_negative` on
    /// a NaN might produce an unexpected or non-portable result. See the [specification
    /// of NaN bit patterns](f32#nan-bit-patterns) for more info. Use `self.signum() == -1.0`
    /// if you need fully portable behavior (will return `false` for all NaNs).
    ///
    /// ```
    /// let f = 7.0_f64;
    /// let g = -7.0_f64;
    ///
    /// assert!(!f.is_sign_negative());
    /// assert!(g.is_sign_negative());
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_classify", since = "1.83.0")]
    #[inline]
    pub const fn is_sign_negative(self) -> bool {
        // IEEE754 says: isSignMinus(x) is true if and only if x has negative sign. isSignMinus
        // applies to zeros and NaNs as well.
        // SAFETY: This is just transmuting to get the sign bit, it's fine.
        unsafe { mem::transmute::<f64, u64>(self) & Self::SIGN_MASK != 0 }
    }

    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[deprecated(since = "1.0.0", note = "renamed to is_sign_negative")]
    #[inline]
    #[doc(hidden)]
    pub fn is_negative(self) -> bool {
        self.is_sign_negative()
    }

    /// Returns the least number greater than `self`.
    ///
    /// Let `TINY` be the smallest representable positive `f64`. Then,
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
    /// #![feature(float_next_up_down)]
    /// // f64::EPSILON is the difference between 1.0 and the next number up.
    /// assert_eq!(1.0f64.next_up(), 1.0 + f64::EPSILON);
    /// // But not for most numbers.
    /// assert!(0.1f64.next_up() < 0.1 + f64::EPSILON);
    /// assert_eq!(9007199254740992f64.next_up(), 9007199254740994.0);
    /// ```
    ///
    /// [`NEG_INFINITY`]: Self::NEG_INFINITY
    /// [`INFINITY`]: Self::INFINITY
    /// [`MIN`]: Self::MIN
    /// [`MAX`]: Self::MAX
    #[inline]
    #[unstable(feature = "float_next_up_down", issue = "91399")]
    pub const fn next_up(self) -> Self {
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
    /// Let `TINY` be the smallest representable positive `f64`. Then,
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
    /// #![feature(float_next_up_down)]
    /// let x = 1.0f64;
    /// // Clamp value into range [0, 1).
    /// let clamped = x.clamp(0.0, 1.0f64.next_down());
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
    pub const fn next_down(self) -> Self {
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
    /// let x = 2.0_f64;
    /// let abs_difference = (x.recip() - (1.0 / x)).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[must_use = "this returns the result of the operation, without modifying the original"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_methods", since = "CURRENT_RUSTC_VERSION")]
    #[inline]
    pub const fn recip(self) -> f64 {
        1.0 / self
    }

    /// Converts radians to degrees.
    ///
    /// ```
    /// let angle = std::f64::consts::PI;
    ///
    /// let abs_difference = (angle.to_degrees() - 180.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_methods", since = "CURRENT_RUSTC_VERSION")]
    #[inline]
    pub const fn to_degrees(self) -> f64 {
        // The division here is correctly rounded with respect to the true
        // value of 180/π. (This differs from f32, where a constant must be
        // used to ensure a correctly rounded result.)
        self * (180.0f64 / consts::PI)
    }

    /// Converts degrees to radians.
    ///
    /// ```
    /// let angle = 180.0_f64;
    ///
    /// let abs_difference = (angle.to_radians() - std::f64::consts::PI).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_methods", since = "CURRENT_RUSTC_VERSION")]
    #[inline]
    pub const fn to_radians(self) -> f64 {
        const RADS_PER_DEG: f64 = consts::PI / 180.0;
        self * RADS_PER_DEG
    }

    /// Returns the maximum of the two numbers, ignoring NaN.
    ///
    /// If one of the arguments is NaN, then the other argument is returned.
    /// This follows the IEEE 754-2008 semantics for maxNum, except for handling of signaling NaNs;
    /// this function handles all NaNs the same way and avoids maxNum's problems with associativity.
    /// This also matches the behavior of libm’s fmax.
    ///
    /// ```
    /// let x = 1.0_f64;
    /// let y = 2.0_f64;
    ///
    /// assert_eq!(x.max(y), y);
    /// ```
    #[must_use = "this returns the result of the comparison, without modifying either input"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_methods", since = "CURRENT_RUSTC_VERSION")]
    #[inline]
    pub const fn max(self, other: f64) -> f64 {
        intrinsics::maxnumf64(self, other)
    }

    /// Returns the minimum of the two numbers, ignoring NaN.
    ///
    /// If one of the arguments is NaN, then the other argument is returned.
    /// This follows the IEEE 754-2008 semantics for minNum, except for handling of signaling NaNs;
    /// this function handles all NaNs the same way and avoids minNum's problems with associativity.
    /// This also matches the behavior of libm’s fmin.
    ///
    /// ```
    /// let x = 1.0_f64;
    /// let y = 2.0_f64;
    ///
    /// assert_eq!(x.min(y), x);
    /// ```
    #[must_use = "this returns the result of the comparison, without modifying either input"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_methods", since = "CURRENT_RUSTC_VERSION")]
    #[inline]
    pub const fn min(self, other: f64) -> f64 {
        intrinsics::minnumf64(self, other)
    }

    /// Returns the maximum of the two numbers, propagating NaN.
    ///
    /// This returns NaN when *either* argument is NaN, as opposed to
    /// [`f64::max`] which only returns NaN when *both* arguments are NaN.
    ///
    /// ```
    /// #![feature(float_minimum_maximum)]
    /// let x = 1.0_f64;
    /// let y = 2.0_f64;
    ///
    /// assert_eq!(x.maximum(y), y);
    /// assert!(x.maximum(f64::NAN).is_nan());
    /// ```
    ///
    /// If one of the arguments is NaN, then NaN is returned. Otherwise this returns the greater
    /// of the two numbers. For this operation, -0.0 is considered to be less than +0.0.
    /// Note that this follows the semantics specified in IEEE 754-2019.
    ///
    /// Also note that "propagation" of NaNs here doesn't necessarily mean that the bitpattern of a NaN
    /// operand is conserved; see the [specification of NaN bit patterns](f32#nan-bit-patterns) for more info.
    #[must_use = "this returns the result of the comparison, without modifying either input"]
    #[unstable(feature = "float_minimum_maximum", issue = "91079")]
    #[inline]
    pub const fn maximum(self, other: f64) -> f64 {
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
    /// [`f64::min`] which only returns NaN when *both* arguments are NaN.
    ///
    /// ```
    /// #![feature(float_minimum_maximum)]
    /// let x = 1.0_f64;
    /// let y = 2.0_f64;
    ///
    /// assert_eq!(x.minimum(y), x);
    /// assert!(x.minimum(f64::NAN).is_nan());
    /// ```
    ///
    /// If one of the arguments is NaN, then NaN is returned. Otherwise this returns the lesser
    /// of the two numbers. For this operation, -0.0 is considered to be less than +0.0.
    /// Note that this follows the semantics specified in IEEE 754-2019.
    ///
    /// Also note that "propagation" of NaNs here doesn't necessarily mean that the bitpattern of a NaN
    /// operand is conserved; see the [specification of NaN bit patterns](f32#nan-bit-patterns) for more info.
    #[must_use = "this returns the result of the comparison, without modifying either input"]
    #[unstable(feature = "float_minimum_maximum", issue = "91079")]
    #[inline]
    pub const fn minimum(self, other: f64) -> f64 {
        if self < other {
            self
        } else if other < self {
            other
        } else if self == other {
            if self.is_sign_negative() && other.is_sign_positive() { self } else { other }
        } else {
            // At least one input is NaN. Use `+` to perform NaN propagation and quieting.
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
    /// assert_eq!(1f64.midpoint(4.0), 2.5);
    /// assert_eq!((-5.5f64).midpoint(8.0), 1.25);
    /// ```
    #[inline]
    #[stable(feature = "num_midpoint", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_const_stable(feature = "num_midpoint", since = "CURRENT_RUSTC_VERSION")]
    pub const fn midpoint(self, other: f64) -> f64 {
        const LO: f64 = f64::MIN_POSITIVE * 2.;
        const HI: f64 = f64::MAX / 2.;

        let (a, b) = (self, other);
        let abs_a = a.abs();
        let abs_b = b.abs();

        if abs_a <= HI && abs_b <= HI {
            // Overflow is impossible
            (a + b) / 2.
        } else if abs_a < LO {
            // Not safe to halve `a` (would underflow)
            a + (b / 2.)
        } else if abs_b < LO {
            // Not safe to halve `b` (would underflow)
            (a / 2.) + b
        } else {
            // Safe to halve `a` and `b`
            (a / 2.) + (b / 2.)
        }
    }

    /// Rounds toward zero and converts to any primitive integer type,
    /// assuming that the value is finite and fits in that type.
    ///
    /// ```
    /// let value = 4.6_f64;
    /// let rounded = unsafe { value.to_int_unchecked::<u16>() };
    /// assert_eq!(rounded, 4);
    ///
    /// let value = -128.9_f64;
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
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[stable(feature = "float_approx_unchecked_to", since = "1.44.0")]
    #[inline]
    pub unsafe fn to_int_unchecked<Int>(self) -> Int
    where
        Self: FloatToInt<Int>,
    {
        // SAFETY: the caller must uphold the safety contract for
        // `FloatToInt::to_int_unchecked`.
        unsafe { FloatToInt::<Int>::to_int_unchecked(self) }
    }

    /// Raw transmutation to `u64`.
    ///
    /// This is currently identical to `transmute::<f64, u64>(self)` on all platforms.
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
    /// assert!((1f64).to_bits() != 1f64 as u64); // to_bits() is not casting!
    /// assert_eq!((12.5f64).to_bits(), 0x4029000000000000);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[stable(feature = "float_bits_conv", since = "1.20.0")]
    #[rustc_const_stable(feature = "const_float_bits_conv", since = "1.83.0")]
    #[inline]
    pub const fn to_bits(self) -> u64 {
        // SAFETY: `u64` is a plain old datatype so we can always transmute to it.
        unsafe { mem::transmute(self) }
    }

    /// Raw transmutation from `u64`.
    ///
    /// This is currently identical to `transmute::<u64, f64>(v)` on all platforms.
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
    /// let v = f64::from_bits(0x4029000000000000);
    /// assert_eq!(v, 12.5);
    /// ```
    #[stable(feature = "float_bits_conv", since = "1.20.0")]
    #[rustc_const_stable(feature = "const_float_bits_conv", since = "1.83.0")]
    #[must_use]
    #[inline]
    pub const fn from_bits(v: u64) -> Self {
        // It turns out the safety issues with sNaN were overblown! Hooray!
        // SAFETY: `u64` is a plain old datatype so we can always transmute from it.
        unsafe { mem::transmute(v) }
    }

    /// Returns the memory representation of this floating point number as a byte array in
    /// big-endian (network) byte order.
    ///
    /// See [`from_bits`](Self::from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// # Examples
    ///
    /// ```
    /// let bytes = 12.5f64.to_be_bytes();
    /// assert_eq!(bytes, [0x40, 0x29, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[stable(feature = "float_to_from_bytes", since = "1.40.0")]
    #[rustc_const_stable(feature = "const_float_bits_conv", since = "1.83.0")]
    #[inline]
    pub const fn to_be_bytes(self) -> [u8; 8] {
        self.to_bits().to_be_bytes()
    }

    /// Returns the memory representation of this floating point number as a byte array in
    /// little-endian byte order.
    ///
    /// See [`from_bits`](Self::from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// # Examples
    ///
    /// ```
    /// let bytes = 12.5f64.to_le_bytes();
    /// assert_eq!(bytes, [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x29, 0x40]);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[stable(feature = "float_to_from_bytes", since = "1.40.0")]
    #[rustc_const_stable(feature = "const_float_bits_conv", since = "1.83.0")]
    #[inline]
    pub const fn to_le_bytes(self) -> [u8; 8] {
        self.to_bits().to_le_bytes()
    }

    /// Returns the memory representation of this floating point number as a byte array in
    /// native byte order.
    ///
    /// As the target platform's native endianness is used, portable code
    /// should use [`to_be_bytes`] or [`to_le_bytes`], as appropriate, instead.
    ///
    /// [`to_be_bytes`]: f64::to_be_bytes
    /// [`to_le_bytes`]: f64::to_le_bytes
    ///
    /// See [`from_bits`](Self::from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// # Examples
    ///
    /// ```
    /// let bytes = 12.5f64.to_ne_bytes();
    /// assert_eq!(
    ///     bytes,
    ///     if cfg!(target_endian = "big") {
    ///         [0x40, 0x29, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    ///     } else {
    ///         [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x29, 0x40]
    ///     }
    /// );
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[stable(feature = "float_to_from_bytes", since = "1.40.0")]
    #[rustc_const_stable(feature = "const_float_bits_conv", since = "1.83.0")]
    #[inline]
    pub const fn to_ne_bytes(self) -> [u8; 8] {
        self.to_bits().to_ne_bytes()
    }

    /// Creates a floating point value from its representation as a byte array in big endian.
    ///
    /// See [`from_bits`](Self::from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// # Examples
    ///
    /// ```
    /// let value = f64::from_be_bytes([0x40, 0x29, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    /// assert_eq!(value, 12.5);
    /// ```
    #[stable(feature = "float_to_from_bytes", since = "1.40.0")]
    #[rustc_const_stable(feature = "const_float_bits_conv", since = "1.83.0")]
    #[must_use]
    #[inline]
    pub const fn from_be_bytes(bytes: [u8; 8]) -> Self {
        Self::from_bits(u64::from_be_bytes(bytes))
    }

    /// Creates a floating point value from its representation as a byte array in little endian.
    ///
    /// See [`from_bits`](Self::from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// # Examples
    ///
    /// ```
    /// let value = f64::from_le_bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x29, 0x40]);
    /// assert_eq!(value, 12.5);
    /// ```
    #[stable(feature = "float_to_from_bytes", since = "1.40.0")]
    #[rustc_const_stable(feature = "const_float_bits_conv", since = "1.83.0")]
    #[must_use]
    #[inline]
    pub const fn from_le_bytes(bytes: [u8; 8]) -> Self {
        Self::from_bits(u64::from_le_bytes(bytes))
    }

    /// Creates a floating point value from its representation as a byte array in native endian.
    ///
    /// As the target platform's native endianness is used, portable code
    /// likely wants to use [`from_be_bytes`] or [`from_le_bytes`], as
    /// appropriate instead.
    ///
    /// [`from_be_bytes`]: f64::from_be_bytes
    /// [`from_le_bytes`]: f64::from_le_bytes
    ///
    /// See [`from_bits`](Self::from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// # Examples
    ///
    /// ```
    /// let value = f64::from_ne_bytes(if cfg!(target_endian = "big") {
    ///     [0x40, 0x29, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    /// } else {
    ///     [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x29, 0x40]
    /// });
    /// assert_eq!(value, 12.5);
    /// ```
    #[stable(feature = "float_to_from_bytes", since = "1.40.0")]
    #[rustc_const_stable(feature = "const_float_bits_conv", since = "1.83.0")]
    #[must_use]
    #[inline]
    pub const fn from_ne_bytes(bytes: [u8; 8]) -> Self {
        Self::from_bits(u64::from_ne_bytes(bytes))
    }

    /// Returns the ordering between `self` and `other`.
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
    /// [`PartialOrd`] and [`PartialEq`] implementations of `f64`. For example,
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
    /// struct GoodBoy {
    ///     name: String,
    ///     weight: f64,
    /// }
    ///
    /// let mut bois = vec![
    ///     GoodBoy { name: "Pucci".to_owned(), weight: 0.1 },
    ///     GoodBoy { name: "Woofer".to_owned(), weight: 99.0 },
    ///     GoodBoy { name: "Yapper".to_owned(), weight: 10.0 },
    ///     GoodBoy { name: "Chonk".to_owned(), weight: f64::INFINITY },
    ///     GoodBoy { name: "Abs. Unit".to_owned(), weight: f64::NAN },
    ///     GoodBoy { name: "Floaty".to_owned(), weight: -5.0 },
    /// ];
    ///
    /// bois.sort_by(|a, b| a.weight.total_cmp(&b.weight));
    ///
    /// // `f64::NAN` could be positive or negative, which will affect the sort order.
    /// if f64::NAN.is_sign_negative() {
    ///     assert!(bois.into_iter().map(|b| b.weight)
    ///         .zip([f64::NAN, -5.0, 0.1, 10.0, 99.0, f64::INFINITY].iter())
    ///         .all(|(a, b)| a.to_bits() == b.to_bits()))
    /// } else {
    ///     assert!(bois.into_iter().map(|b| b.weight)
    ///         .zip([-5.0, 0.1, 10.0, 99.0, f64::INFINITY, f64::NAN].iter())
    ///         .all(|(a, b)| a.to_bits() == b.to_bits()))
    /// }
    /// ```
    #[stable(feature = "total_cmp", since = "1.62.0")]
    #[must_use]
    #[inline]
    pub fn total_cmp(&self, other: &Self) -> crate::cmp::Ordering {
        let mut left = self.to_bits() as i64;
        let mut right = other.to_bits() as i64;

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
        left ^= (((left >> 63) as u64) >> 1) as i64;
        right ^= (((right >> 63) as u64) >> 1) as i64;

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
    /// assert!((-3.0f64).clamp(-2.0, 1.0) == -2.0);
    /// assert!((0.0f64).clamp(-2.0, 1.0) == 0.0);
    /// assert!((2.0f64).clamp(-2.0, 1.0) == 1.0);
    /// assert!((f64::NAN).clamp(-2.0, 1.0).is_nan());
    /// ```
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "clamp", since = "1.50.0")]
    #[rustc_const_stable(feature = "const_float_methods", since = "CURRENT_RUSTC_VERSION")]
    #[inline]
    pub const fn clamp(mut self, min: f64, max: f64) -> f64 {
        const_assert!(
            min <= max,
            "min > max, or either was NaN",
            "min > max, or either was NaN. min = {min:?}, max = {max:?}",
            min: f64,
            max: f64,
        );

        if self < min {
            self = min;
        }
        if self > max {
            self = max;
        }
        self
    }

    /// Computes the absolute value of `self`.
    ///
    /// This function always returns the precise result.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 3.5_f64;
    /// let y = -3.5_f64;
    ///
    /// assert_eq!(x.abs(), x);
    /// assert_eq!(y.abs(), -y);
    ///
    /// assert!(f64::NAN.abs().is_nan());
    /// ```
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_methods", since = "CURRENT_RUSTC_VERSION")]
    #[inline]
    pub const fn abs(self) -> f64 {
        // SAFETY: this is actually a safe intrinsic
        unsafe { intrinsics::fabsf64(self) }
    }

    /// Returns a number that represents the sign of `self`.
    ///
    /// - `1.0` if the number is positive, `+0.0` or `INFINITY`
    /// - `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
    /// - NaN if the number is NaN
    ///
    /// # Examples
    ///
    /// ```
    /// let f = 3.5_f64;
    ///
    /// assert_eq!(f.signum(), 1.0);
    /// assert_eq!(f64::NEG_INFINITY.signum(), -1.0);
    ///
    /// assert!(f64::NAN.signum().is_nan());
    /// ```
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_float_methods", since = "CURRENT_RUSTC_VERSION")]
    #[inline]
    pub const fn signum(self) -> f64 {
        if self.is_nan() { Self::NAN } else { 1.0_f64.copysign(self) }
    }

    /// Returns a number composed of the magnitude of `self` and the sign of
    /// `sign`.
    ///
    /// Equal to `self` if the sign of `self` and `sign` are the same, otherwise equal to `-self`.
    /// If `self` is a NaN, then a NaN with the same payload as `self` and the sign bit of `sign` is
    /// returned.
    ///
    /// If `sign` is a NaN, then this operation will still carry over its sign into the result. Note
    /// that IEEE 754 doesn't assign any meaning to the sign bit in case of a NaN, and as Rust
    /// doesn't guarantee that the bit pattern of NaNs are conserved over arithmetic operations, the
    /// result of `copysign` with `sign` being a NaN might produce an unexpected or non-portable
    /// result. See the [specification of NaN bit patterns](primitive@f32#nan-bit-patterns) for more
    /// info.
    ///
    /// # Examples
    ///
    /// ```
    /// let f = 3.5_f64;
    ///
    /// assert_eq!(f.copysign(0.42), 3.5_f64);
    /// assert_eq!(f.copysign(-0.42), -3.5_f64);
    /// assert_eq!((-f).copysign(0.42), 3.5_f64);
    /// assert_eq!((-f).copysign(-0.42), -3.5_f64);
    ///
    /// assert!(f64::NAN.copysign(1.0).is_nan());
    /// ```
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[stable(feature = "copysign", since = "1.35.0")]
    #[rustc_const_stable(feature = "const_float_methods", since = "CURRENT_RUSTC_VERSION")]
    #[inline]
    pub const fn copysign(self, sign: f64) -> f64 {
        // SAFETY: this is actually a safe intrinsic
        unsafe { intrinsics::copysignf64(self, sign) }
    }
}
