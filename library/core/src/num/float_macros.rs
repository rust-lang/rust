#[rustc_macro_transparency = "semiopaque"]
pub(crate) macro float_impl(
    Self = $SelfT:ty,
    Bits = $BitsT:ty,
    SignedBits = $SignedBitsT:ty,
    // Constants
    BITS = $BITS:literal,
    MANTISSA_DIGITS = $MANTISSA_DIGITS:literal,
    DIGITS = $DIGITS:literal,
    EPSILON = $EPSILON:literal,
    MIN = $MIN:literal,
    MIN_POSITIVE = $MIN_POSITIVE:literal,
    MAX = $MAX:literal,
    MIN_EXP = $MIN_EXP:literal,
    MAX_EXP = $MAX_EXP:literal,
    MIN_10_EXP = $MIN_10_EXP:literal,
    MAX_10_EXP = $MAX_10_EXP:literal,
    SIGN_MASK = $SIGN_MASK:literal,
    EXPONENT_MASK = $EXPONENT_MASK:literal,
    MANTISSA_MASK = $MANTISSA_MASK:literal,
    // Values for docs
    exponent_bias = $exponent_bias:literal,
    lower_than_min = $lower_than_min:literal,
    // Feature gates
    base = #[$base:meta],
    assoc_int_consts = #[$assoc_int_consts:meta],
    is_subnormal = #[$is_subnormal:meta],
    // Const feature gates
    $(unstable_const = #[$unstable_const:meta],)?
    $(const_float_classify = #[$const_float_classify:meta],)?
    // Things needed for `f16`/`f128` doctests
    $(feature = $feature:ident,)?
    $(has_reliable_cfg = $has_reliable_cfg:ident,)?
    $(has_reliable_math_cfg = $has_reliable_math_cfg:ident,)?
) {

    /// The radix or base of the internal representation of
    #[doc = concat!("`", stringify!($SelfT), "`.")]
    #[$assoc_int_consts]
    pub const RADIX: u32 = 2;

    /// The size of this float type in bits.
    #[unstable(feature = "float_bits_const", issue = "151073")]
    pub const BITS: u32 = $BITS;

    /// Number of significant digits in base 2.
    ///
    /// Note that the size of the mantissa in the bitwise representation is one
    /// smaller than this since the leading 1 is not stored explicitly.
    #[$assoc_int_consts]
    pub const MANTISSA_DIGITS: u32 = $MANTISSA_DIGITS;

    /// Approximate number of significant digits in base 10.
    ///
    /// This is the maximum <i>x</i> such that any decimal number with <i>x</i>
    /// significant digits can be converted to
    #[doc = concat!("`", stringify!($SelfT), "`")]
    /// and back without loss.
    ///
    /// Equal to floor(log<sub>10</sub>&nbsp;2<sup>[`MANTISSA_DIGITS`]&nbsp;&minus;&nbsp;1</sup>).
    ///
    /// [`MANTISSA_DIGITS`]: Self::MANTISSA_DIGITS
    #[$assoc_int_consts]
    pub const DIGITS: u32 = $DIGITS;

    /// [Machine epsilon] value for
    #[doc = concat!("`", stringify!($SelfT), "`.")]
    ///
    /// This is the difference between `1.0` and the next larger representable number.
    ///
    /// Equal to 2<sup>1&nbsp;&minus;&nbsp;[`MANTISSA_DIGITS`]</sup>.
    ///
    /// [Machine epsilon]: https://en.wikipedia.org/wiki/Machine_epsilon
    /// [`MANTISSA_DIGITS`]: Self::MANTISSA_DIGITS
    #[$assoc_int_consts]
    #[rustc_diagnostic_item = concat!(stringify!($SelfT), "_epsilon")]
    pub const EPSILON: Self = $EPSILON;

    /// Smallest finite
    #[doc = concat!("`", stringify!($SelfT), "`")]
    /// value.
    ///
    /// Equal to &minus;[`MAX`].
    ///
    /// [`MAX`]: Self::MAX
    #[$assoc_int_consts]
    pub const MIN: Self = $MIN;
    /// Smallest positive normal
    #[doc = concat!("`", stringify!($SelfT), "`")]
    /// value.
    ///
    /// Equal to 2<sup>[`MIN_EXP`]&nbsp;&minus;&nbsp;1</sup>.
    ///
    /// [`MIN_EXP`]: Self::MIN_EXP
    #[$assoc_int_consts]
    pub const MIN_POSITIVE: Self = $MIN_POSITIVE;
    /// Largest finite
    #[doc = concat!("`", stringify!($SelfT), "`")]
    /// value.
    ///
    /// Equal to
    /// (1&nbsp;&minus;&nbsp;2<sup>&minus;[`MANTISSA_DIGITS`]</sup>)&nbsp;2<sup>[`MAX_EXP`]</sup>.
    ///
    /// [`MANTISSA_DIGITS`]: Self::MANTISSA_DIGITS
    /// [`MAX_EXP`]: Self::MAX_EXP
    #[$assoc_int_consts]
    pub const MAX: Self = $MAX;

    /// One greater than the minimum possible *normal* power of 2 exponent
    /// for a significand bounded by 1 ≤ x < 2 (i.e. the IEEE definition).
    ///
    /// This corresponds to the exact minimum possible *normal* power of 2 exponent
    /// for a significand bounded by 0.5 ≤ x < 1 (i.e. the C definition).
    /// In other words, all normal numbers representable by this type are
    /// greater than or equal to 0.5&nbsp;×&nbsp;2<sup><i>MIN_EXP</i></sup>.
    #[$assoc_int_consts]
    pub const MIN_EXP: i32 = $MIN_EXP;
    /// One greater than the maximum possible power of 2 exponent
    /// for a significand bounded by 1 ≤ x < 2 (i.e. the IEEE definition).
    ///
    /// This corresponds to the exact maximum possible power of 2 exponent
    /// for a significand bounded by 0.5 ≤ x < 1 (i.e. the C definition).
    /// In other words, all numbers representable by this type are
    /// strictly less than 2<sup><i>MAX_EXP</i></sup>.
    #[$assoc_int_consts]
    pub const MAX_EXP: i32 = $MAX_EXP;

    /// Minimum <i>x</i> for which 10<sup><i>x</i></sup> is normal.
    ///
    /// Equal to ceil(log<sub>10</sub>&nbsp;[`MIN_POSITIVE`]).
    ///
    /// [`MIN_POSITIVE`]: Self::MIN_POSITIVE
    #[$assoc_int_consts]
    pub const MIN_10_EXP: i32 = $MIN_10_EXP;
    /// Maximum <i>x</i> for which 10<sup><i>x</i></sup> is normal.
    ///
    /// Equal to floor(log<sub>10</sub>&nbsp;[`MAX`]).
    ///
    /// [`MAX`]: Self::MAX
    #[$assoc_int_consts]
    pub const MAX_10_EXP: i32 = $MAX_10_EXP;

    /// Not a Number (NaN).
    ///
    /// Note that IEEE 754 doesn't define just a single NaN value; a plethora of bit patterns are
    /// considered to be NaN. Furthermore, the standard makes a difference between a "signaling" and
    /// a "quiet" NaN, and allows inspecting its "payload" (the unspecified bits in the bit pattern)
    /// and its sign. See the [specification of NaN bit patterns](f32#nan-bit-patterns) for more
    /// info.
    ///
    /// This constant is guaranteed to be a quiet NaN (on targets that follow the Rust assumptions
    /// that the quiet/signaling bit being set to 1 indicates a quiet NaN). Beyond that, nothing is
    /// guaranteed about the specific bit pattern chosen here: both payload and sign are arbitrary.
    /// The concrete bit pattern may change across Rust versions and target platforms.
    #[$assoc_int_consts]
    #[rustc_diagnostic_item = concat!(stringify!($SelfT), "_nan")]
    #[allow(clippy::eq_op, clippy::zero_divided_by_zero)]
    pub const NAN: Self = 0.0 / 0.0;
    /// Infinity (∞).
    #[$assoc_int_consts]
    pub const INFINITY: Self = 1.0 / 0.0;
    /// Negative infinity (−∞).
    #[$assoc_int_consts]
    pub const NEG_INFINITY: Self = -1.0 / 0.0;

    /// Maximum integer that can be represented exactly in an
    #[doc = concat!("[`", stringify!($SelfT), "`]")]
    /// value, with no other integer converting to the same floating point value.
    ///
    /// For an integer `x` which satisfies `MIN_EXACT_INTEGER <= x <= MAX_EXACT_INTEGER`,
    /// there is a "one-to-one" mapping between
    #[doc = concat!(" [`", stringify!($SignedBitsT), "`]")]
    /// and
    #[doc = concat!("[`", stringify!($SelfT), "`]")]
    /// values. `MAX_EXACT_INTEGER + 1` also converts losslessly to
    #[doc = concat!("[`", stringify!($SelfT), "`]")]
    /// and back to
    #[doc = concat!("[`", stringify!($SignedBitsT), "`],")]
    /// but `MAX_EXACT_INTEGER + 2` converts to the same
    #[doc = concat!("[`", stringify!($SelfT), "`]")]
    /// value (and back to `MAX_EXACT_INTEGER + 1` as an integer) so there is not a
    /// "one-to-one" mapping.
    ///
    /// [`MAX_EXACT_INTEGER`]: Self::MAX_EXACT_INTEGER
    /// [`MIN_EXACT_INTEGER`]: Self::MIN_EXACT_INTEGER
    /// ```
    $(#[doc = concat!("#![feature(", stringify!($feature), ")]")])?
    /// #![feature(float_exact_integer_constants)]
    /// # // FIXME(#152635): Float rounding on `i586` does not adhere to IEEE 754
    /// # #[cfg(not(all(target_arch = "x86", not(target_feature = "sse"))))] {
    $(#[doc = concat!("# #[cfg(", stringify!($has_reliable_cfg), ")] {")])?
    #[doc = concat!("let max_exact_int = ", stringify!($SelfT), "::MAX_EXACT_INTEGER;")]
    #[doc = concat!("assert_eq!(max_exact_int, max_exact_int as ", stringify!($SelfT), " as ", stringify!($SignedBitsT), ");")]
    #[doc = concat!("assert_eq!(max_exact_int + 1, (max_exact_int + 1) as ", stringify!($SelfT), " as ", stringify!($SignedBitsT), ");")]
    #[doc = concat!("assert_ne!(max_exact_int + 2, (max_exact_int + 2) as ", stringify!($SelfT), " as ", stringify!($SignedBitsT), ");")]
    ///
    #[doc = concat!("// Beyond `", stringify!($SelfT), "::MAX_EXACT_INTEGER`, multiple integers can map to one float value")]
    #[doc = concat!("assert_eq!((max_exact_int + 1) as ", stringify!($SelfT), ", (max_exact_int + 2) as ", stringify!($SelfT), ");")]
    #[doc = concat!("# }" $(${ignore($has_reliable_cfg)}, "}")?)]
    /// ```
    #[unstable(feature = "float_exact_integer_constants", issue = "152466")]
    pub const MAX_EXACT_INTEGER: $SignedBitsT = (1 << Self::MANTISSA_DIGITS) - 1;

    /// Minimum integer that can be represented exactly in an
    #[doc = concat!("[`", stringify!($SelfT), "`]")]
    /// value, with no other integer converting to the same floating point value.
    ///
    /// For an integer `x` which satisfies `MIN_EXACT_INTEGER <= x <= MAX_EXACT_INTEGER`,
    /// there is a "one-to-one" mapping between
    #[doc = concat!(" [`", stringify!($SignedBitsT), "`]")]
    /// and
    #[doc = concat!("[`", stringify!($SelfT), "`]")]
    /// values. `MAX_EXACT_INTEGER + 1` also converts losslessly to
    #[doc = concat!("[`", stringify!($SelfT), "`]")]
    /// and back to
    #[doc = concat!(" [`", stringify!($SignedBitsT), "`],")]
    /// but `MAX_EXACT_INTEGER + 2` converts to the same
    #[doc = concat!("[`", stringify!($SelfT), "`]")]
    /// value (and back to `MAX_EXACT_INTEGER + 1` as an integer) so there is not a
    /// "one-to-one" mapping.
    ///
    /// This constant is equivalent to `-MAX_EXACT_INTEGER`.
    ///
    /// [`MAX_EXACT_INTEGER`]: Self::MAX_EXACT_INTEGER
    /// [`MIN_EXACT_INTEGER`]: Self::MIN_EXACT_INTEGER
    /// ```
    $(#[doc = concat!("#![feature(", stringify!($feature), ")]")])?
    /// #![feature(float_exact_integer_constants)]
    /// # // FIXME(#152635): Float rounding on `i586` does not adhere to IEEE 754
    /// # #[cfg(not(all(target_arch = "x86", not(target_feature = "sse"))))] {
    $(#[doc = concat!("# #[cfg(", stringify!($has_reliable_cfg), ")] {")])?
    #[doc = concat!("let min_exact_int = ", stringify!($SelfT), "::MIN_EXACT_INTEGER;")]
    #[doc = concat!("assert_eq!(min_exact_int, min_exact_int as ", stringify!($SelfT), " as ", stringify!($SignedBitsT), ");")]
    #[doc = concat!("assert_eq!(min_exact_int - 1, (min_exact_int - 1) as ", stringify!($SelfT), " as ", stringify!($SignedBitsT), ");")]
    #[doc = concat!("assert_ne!(min_exact_int - 2, (min_exact_int - 2) as ", stringify!($SelfT), " as ", stringify!($SignedBitsT), ");")]
    ///
    #[doc = concat!("// Below `", stringify!($SelfT), "::MIN_EXACT_INTEGER`, multiple integers can map to one float value")]
    #[doc = concat!("assert_eq!((min_exact_int - 1) as ", stringify!($SelfT), ", (min_exact_int - 2) as ", stringify!($SelfT), ");")]
    #[doc = concat!("# }" $(${ignore($has_reliable_cfg)}, "}")?)]
    /// ```
    #[unstable(feature = "float_exact_integer_constants", issue = "152466")]
    pub const MIN_EXACT_INTEGER: $SignedBitsT = -Self::MAX_EXACT_INTEGER;

    /// The mask of the bit used to encode the sign of an
    #[doc = concat!("[`", stringify!($SelfT), "`].")]
    ///
    /// This bit is set when the sign is negative and unset when the sign is
    /// positive.
    /// If you only need to check whether a value is positive or negative,
    /// [`is_sign_positive`] or [`is_sign_negative`] can be used.
    ///
    /// [`is_sign_positive`]: Self::is_sign_positive
    /// [`is_sign_negative`]: Self::is_sign_negative
    /// ```
    /// #![feature(float_masks)]
    $(#[doc = concat!("#![feature(", stringify!($feature), ")]")])?
    $(#[doc = concat!("# #[cfg(", stringify!($has_reliable_cfg), ")] {")])?
    #[doc = concat!("let sign_mask = ", stringify!($SelfT), "::SIGN_MASK;")]
    #[doc = concat!("let a = 1.6552", stringify!($SelfT), ";")]
    /// let a_bits = a.to_bits();
    ///
    /// assert_eq!(a_bits & sign_mask, 0x0);
    #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(a_bits ^ sign_mask), -a);")]
    #[doc = concat!("assert_eq!(sign_mask, (-0.0", stringify!($SelfT), ").to_bits());")]
    $(${ignore($has_reliable_cfg)}
        /// # }
    )?
    /// ```
    #[unstable(feature = "float_masks", issue = "154064")]
    pub const SIGN_MASK: $BitsT = $SIGN_MASK;

    /// The mask of the bits used to encode the exponent of an
    #[doc = concat!("[`", stringify!($SelfT), "`].")]
    ///
    /// Note that the exponent is stored as a biased value, with a bias of
    #[doc = stringify!($exponent_bias)]
    /// for
    #[doc = concat!("`", stringify!($SelfT), "`.")]
    ///
    /// ```
    /// #![feature(float_masks)]
    $(#[doc = concat!("#![feature(", stringify!($feature), ")]")])?
    $(#[doc = concat!("# #[cfg(", stringify!($has_reliable_cfg), ")] {")])?
    #[doc = concat!("fn get_exp(a: ", stringify!($SelfT), ") -> ", stringify!($SignedBitsT), " {")]
    #[doc = concat!("    let bias = ", $exponent_bias, ";")]
    #[doc = concat!("    let biased = a.to_bits() & ", stringify!($SelfT), "::EXPONENT_MASK;")]
    #[doc = concat!("    (biased >> (", stringify!($SelfT), "::MANTISSA_DIGITS - 1)).cast_signed() - bias")]
    /// }
    ///
    /// assert_eq!(get_exp(0.5), -1);
    /// assert_eq!(get_exp(1.0), 0);
    /// assert_eq!(get_exp(2.0), 1);
    /// assert_eq!(get_exp(4.0), 2);
    $(${ignore($has_reliable_cfg)}
        /// # }
    )?
    /// ```
    #[unstable(feature = "float_masks", issue = "154064")]
    pub const EXPONENT_MASK: $BitsT = $EXPONENT_MASK;

    /// The mask of the bits used to encode the mantissa of an
    #[doc = concat!("[`", stringify!($SelfT), "`].")]
    ///
    /// ```rust
    /// #![feature(float_masks)]
    $(#[doc = concat!("#![feature(", stringify!($feature), ")]")])?
    $(#[doc = concat!("# #[cfg(", stringify!($has_reliable_cfg), ")] {")])?
    #[doc = concat!("let mantissa_mask = ", stringify!($SelfT), "::MANTISSA_MASK;")]
    ///
    #[doc = concat!("assert_eq!(0", stringify!($SelfT), ".to_bits() & mantissa_mask, 0x0);")]
    #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".to_bits() & mantissa_mask, 0x0);")]
    ///
    /// // multiplying a finite value by a power of 2 doesn't change its mantissa
    /// // unless the result or initial value is not normal.
    #[doc = concat!("let a = 1.6552", stringify!($SelfT), ";")]
    /// let b = 4.0 * a;
    /// assert_eq!(a.to_bits() & mantissa_mask, b.to_bits() & mantissa_mask);
    ///
    /// // The maximum and minimum values have a saturated significand
    #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.to_bits() & ", stringify!($SelfT), "::MANTISSA_MASK, ", stringify!($SelfT), "::MANTISSA_MASK);")]
    #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.to_bits() & ", stringify!($SelfT), "::MANTISSA_MASK, ", stringify!($SelfT), "::MANTISSA_MASK);")]
    $(${ignore($has_reliable_cfg)}
        /// # }
    )?
    /// ```
    #[unstable(feature = "float_masks", issue = "154064")]
    pub const MANTISSA_MASK: $BitsT = $MANTISSA_MASK;

    /// Minimum representable positive value (min subnormal)
    const TINY_BITS: $BitsT = 0x1;

    /// Minimum representable negative value (min negative subnormal)
    const NEG_TINY_BITS: $BitsT = Self::TINY_BITS | Self::SIGN_MASK;

    /// Returns `true` if this value is NaN.
    ///
    /// ```
    $(#[doc = concat!("#![feature(", stringify!($feature), ")]")])?
    $(#[doc = concat!("# #[cfg(", stringify!($has_reliable_cfg), ")] {")])?
    $(${ignore($feature)}
        ///
    )?
    #[doc = concat!("let nan = ", stringify!($SelfT), "::NAN;")]
    #[doc = concat!("let f = 7.0_", stringify!($SelfT), ";")]
    ///
    /// assert!(nan.is_nan());
    /// assert!(!f.is_nan());
    $(${ignore($has_reliable_cfg)}
        /// # }
    )?
    /// ```
    #[inline]
    #[must_use]
    #[$base]
    $(#[$const_float_classify])?
    #[allow(clippy::eq_op)] // > if you intended to check if the operand is NaN, use `.is_nan()` instead :)
    pub const fn is_nan(self) -> bool {
        self != self
    }

    /// Returns `true` if this value is positive infinity or negative infinity, and
    /// `false` otherwise.
    ///
    /// ```
    $(#[doc = concat!("#![feature(", stringify!($feature), ")]")])?
    $(#[doc = concat!("# #[cfg(", stringify!($has_reliable_cfg), ")] {")])?
    $(${ignore($feature)}
        ///
    )?
    #[doc = concat!("let f = 7.0", stringify!($SelfT), ";")]
    #[doc = concat!("let inf = ", stringify!($SelfT), "::INFINITY;")]
    #[doc = concat!("let neg_inf = ", stringify!($SelfT), "::NEG_INFINITY;")]
    #[doc = concat!("let nan = ", stringify!($SelfT), "::NAN;")]
    ///
    /// assert!(!f.is_infinite());
    /// assert!(!nan.is_infinite());
    ///
    /// assert!(inf.is_infinite());
    /// assert!(neg_inf.is_infinite());
    $(${ignore($has_reliable_cfg)}
        /// # }
    )?
    /// ```
    #[inline]
    #[must_use]
    #[$base]
    $(#[$const_float_classify])?
    pub const fn is_infinite(self) -> bool {
        // Getting clever with transmutation can affect results on 32-bit x86 without SSE2 due
        // to unsound miscompilations: see https://github.com/rust-lang/rust/issues/114479
        (self == Self::INFINITY) | (self == Self::NEG_INFINITY)
    }

    /// Returns `true` if this number is neither infinite nor NaN.
    ///
    /// ```
    $(#[doc = concat!("#![feature(", stringify!($feature), ")]")])?
    $(#[doc = concat!("# #[cfg(", stringify!($has_reliable_cfg), ")] {")])?
    $(${ignore($feature)}
        ///
    )?
    #[doc = concat!("let f = 7.0", stringify!($SelfT), ";")]
    #[doc = concat!("let inf = ", stringify!($SelfT), "::INFINITY;")]
    #[doc = concat!("let neg_inf = ", stringify!($SelfT), "::NEG_INFINITY;")]
    #[doc = concat!("let nan = ", stringify!($SelfT), "::NAN;")]
    ///
    /// assert!(f.is_finite());
    ///
    /// assert!(!nan.is_finite());
    /// assert!(!inf.is_finite());
    /// assert!(!neg_inf.is_finite());
    $(${ignore($has_reliable_cfg)}
        /// # }
    )?
    /// ```
    #[inline]
    #[must_use]
    #[$base]
    $(#[$const_float_classify])?
    $(#[$unstable_const])?
    pub const fn is_finite(self) -> bool {
        // There's no need to handle NaN separately: if self is NaN,
        // the comparison is not true, exactly as desired.
        self.abs() < Self::INFINITY
    }

    /// Returns `true` if the number is [subnormal].
    ///
    /// ```
    $(#[doc = concat!("#![feature(", stringify!($feature), ")]")])?
    $(#[doc = concat!("# #[cfg(", stringify!($has_reliable_cfg), ")] {")])?
    $(${ignore($feature)}
        ///
    )?
    #[doc = concat!("let min = ", stringify!($SelfT), "::MIN_POSITIVE; // ", stringify!($MIN_POSITIVE))]
    #[doc = concat!("let max = ", stringify!($SelfT), "::MAX;")]
    #[doc = concat!("let lower_than_min = ", stringify!($lower_than_min), ";")]
    #[doc = concat!("let zero = 0.0_", stringify!($SelfT), ";")]
    ///
    /// assert!(!min.is_subnormal());
    /// assert!(!max.is_subnormal());
    ///
    /// assert!(!zero.is_subnormal());
    #[doc = concat!("assert!(!", stringify!($SelfT), "::NAN.is_subnormal());")]
    #[doc = concat!("assert!(!", stringify!($SelfT), "::INFINITY.is_subnormal());")]
    /// // Values between `0` and `min` are Subnormal.
    /// assert!(lower_than_min.is_subnormal());
    $(${ignore($has_reliable_cfg)}
        /// # }
    )?
    /// ```
    ///
    /// [subnormal]: https://en.wikipedia.org/wiki/Subnormal_number
    #[inline]
    #[must_use]
    #[$is_subnormal]
    $(#[$const_float_classify])?
    pub const fn is_subnormal(self) -> bool {
        matches!(self.classify(), FpCategory::Subnormal)
    }

    /// Returns `true` if the number is neither zero, infinite, [subnormal], or NaN.
    ///
    /// ```
    $(#[doc = concat!("#![feature(", stringify!($feature), ")]")])?
    $(#[doc = concat!("# #[cfg(", stringify!($has_reliable_cfg), ")] {")])?
    $(${ignore($feature)}
        ///
    )?
    #[doc = concat!("let min = ", stringify!($SelfT), "::MIN_POSITIVE; // ", stringify!($MIN_POSITIVE))]
    #[doc = concat!("let max = ", stringify!($SelfT), "::MAX;")]
    #[doc = concat!("let lower_than_min = ", stringify!($lower_than_min), ";")]
    #[doc = concat!("let zero = 0.0_", stringify!($SelfT), ";")]
    ///
    /// assert!(min.is_normal());
    /// assert!(max.is_normal());
    ///
    /// assert!(!zero.is_normal());
    #[doc = concat!("assert!(!", stringify!($SelfT), "::NAN.is_normal());")]
    #[doc = concat!("assert!(!", stringify!($SelfT), "::INFINITY.is_normal());")]
    /// // Values between `0` and `min` are Subnormal.
    /// assert!(!lower_than_min.is_normal());
    $(${ignore($has_reliable_cfg)}
        /// # }
    )?
    /// ```
    ///
    /// [subnormal]: https://en.wikipedia.org/wiki/Subnormal_number
    #[inline]
    #[must_use]
    #[$base]
    $(#[$const_float_classify])?
    pub const fn is_normal(self) -> bool {
        matches!(self.classify(), FpCategory::Normal)
    }

    /// Returns the floating point category of the number. If only one property
    /// is going to be tested, it is generally faster to use the specific
    /// predicate instead.
    ///
    /// ```
    $(#[doc = concat!("#![feature(", stringify!($feature), ")]")])?
    $(#[doc = concat!("# #[cfg(", stringify!($has_reliable_cfg), ")] {")])?
    $(${ignore($feature)}
        ///
    )?
    /// use std::num::FpCategory;
    ///
    #[doc = concat!("let num = 12.4_", stringify!($SelfT), ";")]
    #[doc = concat!("let inf = ", stringify!($SelfT), "::INFINITY;")]
    ///
    /// assert_eq!(num.classify(), FpCategory::Normal);
    /// assert_eq!(inf.classify(), FpCategory::Infinite);
    $(${ignore($has_reliable_cfg)}
        /// # }
    )?
    /// ```
    #[$base]
    $(#[$const_float_classify])?
    #[must_use]
    pub const fn classify(self) -> FpCategory {
        // We used to have complicated logic here that avoids the simple bit-based tests to work
        // around buggy codegen on x87 targets for `f32` and `f64` (see
        // https://github.com/rust-lang/rust/issues/114479). However, some LLVM versions later,
        // none of our tests is able to find any difference between the complicated and the
        // naive version, so now we are back to the naive version.
        let bits = self.to_bits();
        match (bits & Self::MANTISSA_MASK, bits & Self::EXPONENT_MASK) {
            (0, Self::EXPONENT_MASK) => FpCategory::Infinite,
            (_, Self::EXPONENT_MASK) => FpCategory::Nan,
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
    /// if you need fully portable behavior (evaluates to `false` for all NaNs).
    ///
    /// ```
    $(#[doc = concat!("#![feature(", stringify!($feature), ")]")])?
    $(#[doc = concat!("# #[cfg(", stringify!($has_reliable_cfg), ")] {")])?
    $(${ignore($feature)}
        ///
    )?
    #[doc = concat!("let f = 7.0_", stringify!($SelfT), ";")]
    #[doc = concat!("let g = -7.0_", stringify!($SelfT), ";")]
    ///
    /// assert!(f.is_sign_positive());
    /// assert!(!g.is_sign_positive());
    $(${ignore($has_reliable_cfg)}
        /// # }
    )?
    /// ```
    #[inline]
    #[must_use]
    #[$base]
    $(#[$const_float_classify])?
    pub const fn is_sign_positive(self) -> bool {
        !self.is_sign_negative()
    }

    /// Returns `true` if `self` has a negative sign, including `-0.0`, NaNs with
    /// negative sign bit and negative infinity.
    ///
    /// Note that IEEE 754 doesn't assign any meaning to the sign bit in case of
    /// a NaN, and as Rust doesn't guarantee that the bit pattern of NaNs are
    /// conserved over arithmetic operations, the result of `is_sign_negative` on
    /// a NaN might produce an unexpected or non-portable result. See the [specification
    /// of NaN bit patterns](f32#nan-bit-patterns) for more info. Use `self.signum() == -1.0`
    /// if you need fully portable behavior (evaluates to `false` for all NaNs).
    ///
    /// ```
    $(#[doc = concat!("#![feature(", stringify!($feature), ")]")])?
    $(#[doc = concat!("# #[cfg(", stringify!($has_reliable_cfg), ")] {")])?
    $(${ignore($feature)}
        ///
    )?
    #[doc = concat!("let f = 7.0_", stringify!($SelfT), ";")]
    #[doc = concat!("let g = -7.0_", stringify!($SelfT), ";")]
    ///
    /// assert!(!f.is_sign_negative());
    /// assert!(g.is_sign_negative());
    $(${ignore($has_reliable_cfg)}
        /// # }
    )?
    /// ```
    #[inline]
    #[must_use]
    #[$base]
    $(#[$const_float_classify])?
    pub const fn is_sign_negative(self) -> bool {
        // IEEE754 says: isSignMinus(x) is true if and only if x has negative sign. isSignMinus
        // applies to zeros and NaNs as well.
        (self.to_bits() & Self::SIGN_MASK) != 0
    }
}
