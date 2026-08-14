//! PowerPC intrinsics

pub(crate) mod macros;

/// The IBM extended-precision (double-double) floating-point type.
#[lang = "ppcf128"]
#[doc(alias = "__ibm128")]
#[doc(alias = "doubledouble")]
#[doc(alias = "f64f64")]
#[unstable(feature = "powerpc_ppcf128", issue = "161787")]
#[allow(non_camel_case_types)]
#[doc(cfg(any(target_arch = "powerpc", target_arch = "powerpc64")))]
pub struct ppcf128([u8; 16]);

impl ppcf128 {
    /// The size of this float type in bits.
    #[unstable(feature = "powerpc_ppcf128", issue = "161787")]
    const BITS: u32 = 128;

    /// Smallest finite `ppcf128` value.
    ///
    /// Equal to &minus;`MAX`.
    #[unstable(feature = "powerpc_ppcf128", issue = "161787")]
    pub const MIN: Self =
        unsafe { Self::from_components_unchecked(f64::MIN, f64::MIN * f64::EPSILON / 4.0) };

    /// Largest finite `ppcf128` value.
    #[unstable(feature = "powerpc_ppcf128", issue = "161787")]
    pub const MAX: Self = Self::from_components(f64::MAX, f64::MAX * f64::EPSILON / 4.0);

    /// Not a Number (NaN).
    #[unstable(feature = "powerpc_ppcf128", issue = "161787")]
    pub const NAN: Self = Self::from_components(f64::NAN, 0.0);

    /// Infinity (∞).
    #[unstable(feature = "powerpc_ppcf128", issue = "161787")]
    pub const INFINITY: Self = Self::from_components(f64::INFINITY, 0.0);

    /// Negative infinity (−∞).
    #[unstable(feature = "powerpc_ppcf128", issue = "161787")]
    pub const NEG_INFINITY: Self = Self::from_components(f64::NEG_INFINITY, 0.0);

    /// Returns the memory representation of this floating point number as a byte array in
    /// native byte order.
    #[unstable(feature = "powerpc_ppcf128", issue = "161787")]
    #[inline]
    pub const fn to_ne_bytes(self) -> [u8; 16] {
        // SAFETY: every bit pattern of a `ppcf128` is a valid `[u8; 16]`.
        unsafe { crate::mem::transmute::<ppcf128, [u8; 16]>(self) }
    }

    /// Returns the memory representation of this floating point number as a byte array in
    /// little-endian byte order.
    #[unstable(feature = "powerpc_ppcf128", issue = "none")]
    #[inline]
    pub const fn to_le_bytes(self) -> [u8; 16] {
        let mut bytes = self.to_ne_bytes();
        if cfg!(target_endian = "big") {
            bytes.reverse();
        }
        bytes
    }

    /// Returns the memory representation of this floating point number as a byte array in
    /// big-endian (network) byte order.
    #[unstable(feature = "powerpc_ppcf128", issue = "none")]
    #[inline]
    pub const fn to_be_bytes(self) -> [u8; 16] {
        let mut bytes = self.to_ne_bytes();
        if cfg!(target_endian = "little") {
            bytes.reverse();
        }
        bytes
    }

    /// Check whether the large and small component are in normal form.
    const fn is_normal_form(large: f64, small: f64) -> bool {
        let is_elfv2 = cfg!(all(target_arch = "powerpc64", target_endian = "little"));

        if large.is_nan() {
            true
        } else if large.is_infinite() && is_elfv2 {
            small == 0.0
        } else {
            large + small == large
        }
    }

    /// Create a [`ppcf128`] from its large and small components.
    ///
    /// This function will normalize the components if they are not already in normal form.
    #[unstable(feature = "powerpc_ppcf128", issue = "none")]
    pub const fn from_components(mut large: f64, mut small: f64) -> Self {
        (large, small) = if Self::is_normal_form(large, small) {
            (large, small)
        } else if !(large + small).is_finite() {
            (large, 0.0)
        } else {
            let (x, y) = (large, small);
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

    /// Create a [`ppcf128`] from its large and small components.
    ///
    /// # Safety
    ///
    /// This function is safe to call only when the large and small components are normalized.
    #[unstable(feature = "powerpc_ppcf128", issue = "161787")]
    pub const unsafe fn from_components_unchecked(large: f64, small: f64) -> Self {
        unsafe { core::mem::transmute([large, small]) }
    }

    /// Create a [`ppcf128`] from its large and small components.
    ///
    /// Returns `None` when the components are not in normal form.
    #[unstable(feature = "powerpc_ppcf128", issue = "161787")]
    pub const fn checked_from_components(large: f64, small: f64) -> Option<Self> {
        if Self::is_normal_form(large, small) {
            // SAFETY: the components are in normal form.
            Some(unsafe { Self::from_components_unchecked(large, small) })
        } else {
            None
        }
    }
}

#[unstable(feature = "powerpc_ppcf128", issue = "161787")]
impl Clone for ppcf128 {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
#[unstable(feature = "powerpc_ppcf128", issue = "161787")]
impl Copy for ppcf128 {}

#[unstable(feature = "powerpc_ppcf128", issue = "none")]
impl PartialEq for ppcf128 {
    #[inline]
    fn eq(&self, other: &ppcf128) -> bool {
        *self == *other
    }
}

#[unstable(feature = "powerpc_ppcf128", issue = "161787")]
impl PartialOrd for ppcf128 {
    #[inline]
    fn partial_cmp(&self, other: &ppcf128) -> Option<crate::cmp::Ordering> {
        use crate::cmp::Ordering;
        match ((*self) <= (*other), (*self) >= (*other)) {
            (false, false) => None,
            (false, true) => Some(Ordering::Greater),
            (true, false) => Some(Ordering::Less),
            (true, true) => Some(Ordering::Equal),
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
    crate::intrinsics::abort()
}
