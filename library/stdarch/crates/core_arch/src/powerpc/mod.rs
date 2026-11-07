//! PowerPC intrinsics

pub(crate) mod macros;

macro_rules! ppcf128_doctest_prelude {
    () => {
        concat!(
            "#![feature(powerpc_ppcf128)]\n",
            "# #![feature(stdarch_powerpc)]\n",
            cfg_select! {
                target_arch = "powerpc" => "# use core::arch::powerpc::ppcf128;\n",
                target_arch = "powerpc64" => "# use core::arch::powerpc64::ppcf128;\n",
                _ => "", // just for docs
            },
        )
    };
}

/// The IBM extended-precision (double-double) floating-point type.
///
/// This type consists of two `f64` components, `large` and `small`, with the logical
/// value (in a sufficiently large floating point type) representing `large + small`.
///
/// The intuition is that large `f64` values have less precision behind the decimal dot.
/// Combining a large `f64` with a small one provides meaningful extra precision, while
/// being able to use existing `f64` hardware instructions.
///
/// # Normal form
///
/// The large and small component must be in normal form. Intuitively, the large component must
/// indeed be large, and the small component should only add precision that does not fit in the
/// large component. Creating a value with components that are not in normal form
/// (e.g. with [`ppcf128::from_components_unchecked`]) is [undefined behavior].
///
/// More precisely, `large` and `small` are in normal form if:
///
/// - `large.is_nan()` (`small` is irrelevant when `large.is_nan()`)
/// - `large.is_infinite()` and `small` equals `0.0`
/// - `large + small` equals `large` (as an `f64`)
///
/// The [`ppcf128::from_components`] method automatically normalizes the components.
///
/// ## Prefered and allowed representations
///
/// The table below describes the representations of `ppcf128` values. The preferred representation
/// is the one used by associated constants (if applicable). The allowed representations are those
/// that don't cause [undefined behavior].
///
/// Value | Prefered Representation | Allowed Representations | Notes
/// ---|---|---|---
/// +0.0 | (+0.0, +0.0) | (+0.0, ±0.0) | Nonzero second components are not allowed
/// -0.0 | (-0.0, -0.0) | (-0.0, ±0.0) | Nonzero second components are not allowed
/// +∞ | (+∞, +0.0) | (+∞, ±0.0) | Nonzero second components are not allowed
/// -∞ | (-∞, -0.0) | (-∞, ±0.0) | Nonzero second components are not allowed
/// NaN | ([NaN], +0.0) | (NaN, *) | Any NaN is allowed in the first component, any `f64` is allowed in the second component
/// x ∈ ℝ | (x₁, x₂) | N/A | x₁ + x₂ must round to x₁
///
/// # Endianness
///
/// The `large` and `small` components are always stored as `[large, small]`. This can be
/// unintuitive when treating the bits as a `u128`. The sign bit of the large component
/// is in position 127 on big-endian, but in position 63 on little-endian.
///
/// ```
#[doc = ppcf128_doctest_prelude!()]
///
/// fn to_bits(v: ppcf128) -> u128 {
///     u128::from_ne_bytes(v.to_ne_bytes())
/// }
///
/// if cfg!(target_endian = "little") {
///     assert_eq!(to_bits(ppcf128::from(1.0_f64)),  0x0000_0000_0000_0000_3ff0_0000_0000_0000);
///     assert_eq!(to_bits(ppcf128::from(-1.0_f64)), 0x0000_0000_0000_0000_bff0_0000_0000_0000);
/// } else {
///     assert_eq!(to_bits(ppcf128::from(1.0_f64)),  0x3ff0_0000_0000_0000_0000_0000_0000_0000);
///     assert_eq!(to_bits(ppcf128::from(-1.0_f64)), 0xbff0_0000_0000_0000_0000_0000_0000_0000);
/// }
/// ```
///
/// [NaN]: f64#associatedconstant.NAN
/// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
#[doc(alias = "__ibm128")]
#[doc(alias = "doubledouble")]
#[doc(alias = "f64f64")]
#[unstable(feature = "powerpc_ppcf128", issue = "163323")]
#[allow(non_camel_case_types)]
#[doc(cfg(any(target_arch = "powerpc", target_arch = "powerpc64")))]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct ppcf128(crate::num::imp::ppcf128::ppcf128);

#[unstable(feature = "powerpc_ppcf128", issue = "163323")]
impl crate::fmt::Debug for ppcf128 {
    fn fmt(&self, f: &mut crate::fmt::Formatter<'_>) -> crate::fmt::Result {
        let (hi, lo) = self.to_components();
        write!(f, "ppcf128({hi:?}, {lo:?})")
    }
}

#[unstable(feature = "powerpc_ppcf128", issue = "163323")]
impl Default for ppcf128 {
    fn default() -> Self {
        Self::from_components(0.0, 0.0)
    }
}

#[unstable(feature = "powerpc_ppcf128", issue = "163323")]
impl PartialEq for ppcf128 {
    #[inline]
    fn eq(&self, other: &ppcf128) -> bool {
        self.to_components() == other.to_components()
    }
}

#[unstable(feature = "powerpc_ppcf128", issue = "163323")]
impl PartialOrd for ppcf128 {
    #[inline]
    fn partial_cmp(&self, other: &ppcf128) -> Option<crate::cmp::Ordering> {
        self.to_components().partial_cmp(&other.to_components())
    }
}

#[unstable(feature = "powerpc_ppcf128", issue = "163323")]
impl From<f64> for ppcf128 {
    fn from(x: f64) -> Self {
        // SAFETY: with 0.0 as the low component the value is always in normal form.
        unsafe { ppcf128::from_components_unchecked(x, 0.0) }
    }
}

#[unstable(feature = "powerpc_ppcf128", issue = "163323")]
impl From<f32> for ppcf128 {
    fn from(x: f32) -> Self {
        Self::from(f64::from(x))
    }
}

#[unstable(feature = "powerpc_ppcf128", issue = "163323")]
impl From<f16> for ppcf128 {
    fn from(x: f16) -> Self {
        Self::from(x as f64)
    }
}

impl ppcf128 {
    /// The size of this float type in bits.
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    const BITS: u32 = 128;

    /// Smallest finite `ppcf128` value.
    ///
    /// Equal to &minus;[`MAX`].
    ///
    /// [`MAX`]: ppcf128::MAX
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    pub const MIN: Self =
        unsafe { Self::from_components_unchecked(f64::MIN, f64::MIN * f64::EPSILON / 4.0) };

    /// Largest finite `ppcf128` value.
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    pub const MAX: Self =
        unsafe { Self::from_components_unchecked(f64::MAX, f64::MAX * f64::EPSILON / 4.0) };

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
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    pub const NAN: Self = Self::from_components(f64::NAN, 0.0);

    /// Infinity (∞).
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    pub const INFINITY: Self = Self::from_components(f64::INFINITY, 0.0);

    /// Negative infinity (−∞).
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    pub const NEG_INFINITY: Self = Self::from_components(f64::NEG_INFINITY, 0.0);

    /// Returns the memory representation of this floating point number as a byte array in
    /// native byte order.
    ///
    /// As the target platform's native endianness is used, portable code
    /// should use [`to_be_bytes`] or [`to_le_bytes`], as appropriate, instead.
    ///
    /// [`to_be_bytes`]: ppcf128::to_be_bytes
    /// [`to_le_bytes`]: ppcf128::to_le_bytes
    ///
    /// # Examples
    ///
    /// ```
    #[doc = ppcf128_doctest_prelude!()]
    ///
    /// let bytes = ppcf128::from(12.5_f64).to_ne_bytes();
    /// assert_eq!(
    ///     bytes,
    ///     if cfg!(target_endian = "big") {
    ///         [0x40, 0x29, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ///          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    ///     } else {
    ///         [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x29, 0x40,
    ///          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    ///     }
    /// );
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[rustc_const_unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[inline]
    pub const fn to_ne_bytes(self) -> [u8; 16] {
        // SAFETY: all bytes are initialized.
        unsafe { crate::mem::transmute::<ppcf128, [u8; 16]>(self) }
    }

    /// Returns the memory representation of this floating point number as a byte array in
    /// little-endian byte order.
    ///
    /// # Examples
    ///
    /// ```
    #[doc = ppcf128_doctest_prelude!()]
    ///
    /// let bytes = ppcf128::from(12.5_f64).to_le_bytes();
    /// assert_eq!(
    ///     bytes,
    ///     [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x29, 0x40,
    ///      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    /// );
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[rustc_const_unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[inline]
    pub const fn to_le_bytes(self) -> [u8; 16] {
        let mut bytes = self.to_ne_bytes();
        if cfg!(target_endian = "big") {
            bytes[..8].reverse();
            bytes[8..].reverse();
        }
        bytes
    }

    /// Returns the memory representation of this floating point number as a byte array in
    /// big-endian (network) byte order.
    ///
    /// # Examples
    ///
    /// ```
    #[doc = ppcf128_doctest_prelude!()]
    ///
    /// let bytes = ppcf128::from(12.5_f64).to_be_bytes();
    /// assert_eq!(
    ///     bytes,
    ///     [0x40, 0x29, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ///      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    /// );
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[rustc_const_unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[inline]
    pub const fn to_be_bytes(self) -> [u8; 16] {
        let mut bytes = self.to_ne_bytes();
        if cfg!(target_endian = "little") {
            bytes[..8].reverse();
            bytes[8..].reverse();
        }
        bytes
    }

    /// Returns the large and small component of this floating point number.
    ///
    /// ```
    #[doc = ppcf128_doctest_prelude!()]
    ///
    /// assert_eq!(ppcf128::default().to_components(), (0.0, 0.0));
    /// assert_eq!(ppcf128::INFINITY.to_components(), (f64::INFINITY, 0.0));
    /// assert_eq!(ppcf128::MAX.to_components(), (f64::MAX, f64::MAX * f64::EPSILON / 4.0));
    /// ```
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[rustc_const_unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[inline]
    pub const fn to_components(self) -> (f64, f64) {
        // SAFETY: every bit pattern of a `ppcf128` is a valid `[f64; 2]`.
        let [large, small] = unsafe { crate::mem::transmute::<ppcf128, [f64; 2]>(self) };
        (large, small)
    }

    /// Check whether the large and small component are in normal form.
    const fn is_normal_form(large: f64, small: f64) -> bool {
        if large.is_nan() {
            true
        } else if large.is_infinite() {
            // NOTE: the elfv1 spec does not require this explicitly, but we suspect that
            // it is required implicitly. Without it, infinity is not unique.
            //
            // The elfv2 and 32-bit spec do explicitly require that the low component of
            // infinity is zero.
            small == 0.0
        } else {
            large + small == large
        }
    }

    /// Creates a [`ppcf128`] from its large and small components.
    ///
    /// The result has the intuitive value of `large + small`. This function will normalize
    /// the components if they are not already in [normal form](ppcf128#normal-form).
    ///
    /// ```
    #[doc = ppcf128_doctest_prelude!()]
    ///
    /// assert_eq!(ppcf128::from_components(1.0, 0.0).to_components(), (1.0, 0.0));
    /// assert_eq!(ppcf128::from_components(1.0, 1.0).to_components(), (2.0, 0.0));
    ///
    /// assert_eq!(
    ///     ppcf128::from_components(f64::INFINITY, 1.0).to_components(),
    ///     (f64::INFINITY, 0.0)
    /// );
    /// ```
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    pub const fn from_components(large: f64, small: f64) -> Self {
        let (x, y) = (large, small);
        let (large, small) = if Self::is_normal_form(x, y) {
            (x, y)
        } else if !(x + y).is_finite() {
            (x + y, 0.0)
        } else {
            let large = x + y;
            // Per https://doi.org/10.1145/3121432, Algorithm 2
            let x1 = large - y;
            let y1 = large - x1;
            let x2 = x - x1;
            let y2 = y - y1;
            let small = x2 + y2;
            debug_assert!(Self::is_normal_form(large, small));
            (large, small)
        };

        // SAFETY: the components are in normal form.
        unsafe { Self::from_components_unchecked(large, small) }
    }

    /// Creates a [`ppcf128`] from its large and small components.
    ///
    /// # Safety
    ///
    /// This function is safe to call only when the large and small components are
    /// in [normal form](ppcf128#normal-form).
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    pub const unsafe fn from_components_unchecked(large: f64, small: f64) -> Self {
        crate::assert_unsafe_precondition!(
            check_library_ub,
            "components must be in normal form",
            (large: f64 = large, small: f64 = small) => ppcf128::is_normal_form(large, small),
        );
        unsafe { core::mem::transmute([large, small]) }
    }

    /// Creates a [`ppcf128`] from its large and small components.
    ///
    /// Returns `None` when the components are not in normal form.
    ///
    /// ```
    #[doc = ppcf128_doctest_prelude!()]
    ///
    /// assert!(ppcf128::from_components_checked(1.0, 0.0).is_some());
    /// assert!(ppcf128::from_components_checked(f64::NAN, 0.0).is_some());
    /// assert!(ppcf128::from_components_checked(0.0, 1.0).is_none());
    /// assert!(ppcf128::from_components_checked(0.0, f64::NAN).is_none());
    /// ```
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    pub const fn from_components_checked(large: f64, small: f64) -> Option<Self> {
        if Self::is_normal_form(large, small) {
            // SAFETY: the components are in normal form.
            Some(unsafe { Self::from_components_unchecked(large, small) })
        } else {
            None
        }
    }

    /// Returns `true` if this value is NaN.
    ///
    /// ```
    #[doc = ppcf128_doctest_prelude!()]
    ///
    /// let nan = ppcf128::NAN;
    /// let f = ppcf128::from(7.0_f64);
    ///
    /// assert!(nan.is_nan());
    /// assert!(!f.is_nan());
    /// ```
    #[must_use]
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[rustc_const_unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[inline]
    pub const fn is_nan(self) -> bool {
        self.to_components().0.is_nan()
    }

    /// Returns `true` if this value is positive infinity or negative infinity, and
    /// `false` otherwise.
    ///
    /// ```
    #[doc = ppcf128_doctest_prelude!()]
    ///
    /// let f = ppcf128::from(7.0_f64);
    /// let inf = ppcf128::INFINITY;
    /// let neg_inf = ppcf128::NEG_INFINITY;
    /// let nan = ppcf128::NAN;
    ///
    /// assert!(!f.is_infinite());
    /// assert!(!nan.is_infinite());
    ///
    /// assert!(inf.is_infinite());
    /// assert!(neg_inf.is_infinite());
    /// ```
    #[must_use]
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[rustc_const_unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[inline]
    pub const fn is_infinite(self) -> bool {
        self.to_components().0.is_infinite()
    }

    /// Returns `true` if this number is neither infinite nor NaN.
    ///
    /// ```
    #[doc = ppcf128_doctest_prelude!()]
    ///
    /// let f = ppcf128::from(7.0_f64);
    /// let inf = ppcf128::INFINITY;
    /// let neg_inf = ppcf128::NEG_INFINITY;
    /// let nan = ppcf128::NAN;
    ///
    /// assert!(f.is_finite());
    ///
    /// assert!(!nan.is_finite());
    /// assert!(!inf.is_finite());
    /// assert!(!neg_inf.is_finite());
    /// ```
    #[must_use]
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[rustc_const_unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[inline]
    pub const fn is_finite(self) -> bool {
        self.to_components().0.is_finite()
    }

    /// Returns `true` if `self` has a positive sign, including `+0.0`, NaNs with
    /// positive sign bit and positive infinity.
    ///
    /// Note that IEEE 754 doesn't assign any meaning to the sign bit in case of
    /// a NaN, and as Rust doesn't guarantee that the bit pattern of NaNs are
    /// conserved over arithmetic operations, the result of `is_sign_positive` on
    /// a NaN might produce an unexpected or non-portable result. See the [specification
    /// of NaN bit patterns](f32#nan-bit-patterns) for more info. Use
    /// `self.signum() == ppcf128::from(1.0_f64)` if you need fully portable behavior
    /// (evaluates to `false` for all NaNs).
    ///
    /// ```
    #[doc = ppcf128_doctest_prelude!()]
    ///
    /// let f = ppcf128::from(7.0_f64);
    /// let g = ppcf128::from(-7.0_f64);
    ///
    /// assert!(f.is_sign_positive());
    /// assert!(!g.is_sign_positive());
    /// ```
    #[must_use]
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[rustc_const_unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[inline]
    pub const fn is_sign_positive(self) -> bool {
        self.to_components().0.is_sign_positive()
    }

    /// Returns `true` if `self` has a negative sign, including `-0.0`, NaNs with
    /// negative sign bit and negative infinity.
    ///
    /// Note that IEEE 754 doesn't assign any meaning to the sign bit in case of
    /// a NaN, and as Rust doesn't guarantee that the bit pattern of NaNs are
    /// conserved over arithmetic operations, the result of `is_sign_negative` on
    /// a NaN might produce an unexpected or non-portable result. See the [specification
    /// of NaN bit patterns](f32#nan-bit-patterns) for more info. Use
    /// `self.signum() == ppcf128::from(-1.0_f64)` if you need fully portable behavior
    /// (evaluates to `false` for all NaNs).
    ///
    /// ```
    #[doc = ppcf128_doctest_prelude!()]
    ///
    /// let f = ppcf128::from(7.0_f64);
    /// let g = ppcf128::from(-7.0_f64);
    ///
    /// assert!(!f.is_sign_negative());
    /// assert!(g.is_sign_negative());
    /// ```
    #[must_use]
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[rustc_const_unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[inline]
    pub const fn is_sign_negative(self) -> bool {
        self.to_components().0.is_sign_negative()
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
    #[doc = ppcf128_doctest_prelude!()]
    ///
    /// let f = ppcf128::from(3.5_f64);
    ///
    /// assert_eq!(f.signum(), ppcf128::from(1.0_f64));
    /// assert_eq!(ppcf128::NEG_INFINITY.signum(), ppcf128::from(-1.0_f64));
    ///
    /// assert!(ppcf128::NAN.signum().is_nan());
    /// ```
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[rustc_const_unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[inline]
    pub const fn signum(self) -> Self {
        if self.is_nan() {
            Self::NAN
        } else {
            Self::from_components(1.0_f64.copysign(self.to_components().0), 0.0)
        }
    }
}

mod altivec;
#[unstable(feature = "stdarch_powerpc", issue = "111145")]
pub use self::altivec::*;

mod vsx;
#[unstable(feature = "stdarch_powerpc", issue = "111145")]
pub use self::vsx::*;

#[cfg(test)]
use stdarch_test::assert_instr;

/// Generates the trap instruction `TRAP`
#[cfg_attr(test, assert_instr(trap))]
#[inline]
#[unstable(feature = "stdarch_powerpc", issue = "111145")]
pub unsafe fn trap() -> ! {
    crate::intrinsics::abort_immediate()
}
