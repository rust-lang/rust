//! PowerPC intrinsics

pub(crate) mod macros;

/// The IBM extended-precision (double-double) floating-point type.
// FIXME(ppcf128) improve the docs.
#[lang = "ppcf128"]
#[doc(alias = "__ibm128")]
#[doc(alias = "doubledouble")]
#[doc(alias = "f64f64")]
#[unstable(feature = "powerpc_ppcf128", issue = "163323")]
#[allow(non_camel_case_types)]
#[doc(cfg(any(target_arch = "powerpc", target_arch = "powerpc64")))]
#[derive(Clone, Copy)]
pub struct ppcf128(u128);

#[unstable(feature = "powerpc_ppcf128", issue = "163323")]
impl crate::fmt::Debug for ppcf128 {
    fn fmt(&self, f: &mut crate::fmt::Formatter<'_>) -> crate::fmt::Result {
        let (hi, lo) = self.to_components();
        write!(f, "ppcf128({hi}, {lo})")
    }
}

impl ppcf128 {
    /// Raw transmutation to `u128`.
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[inline]
    pub const fn to_bits(self) -> u128 {
        u128::from_ne_bytes(self.to_ne_bytes())
    }

    /// Raw transmutation from `u128`.
    ///
    /// # Safety
    ///
    /// The bit pattern must represent a valid [`ppcf128`]: the value must be normalized.
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[inline]
    pub const unsafe fn from_bits(bits: u128) -> Self {
        // SAFETY: caller guarantees this is a valid ppcf128 bit pattern.
        unsafe { crate::mem::transmute::<u128, ppcf128>(bits) }
    }

    /// Returns the memory representation of this floating point number as a byte array in
    /// native byte order.
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[inline]
    pub const fn to_ne_bytes(self) -> [u8; 16] {
        // SAFETY: every bit pattern of a `ppcf128` is a valid `[u8; 16]`.
        unsafe { crate::mem::transmute::<ppcf128, [u8; 16]>(self) }
    }

    /// Returns the large and small component of this floating point number.
    #[unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[rustc_const_unstable(feature = "powerpc_ppcf128", issue = "163323")]
    #[inline]
    pub const fn to_components(self) -> (f64, f64) {
        // SAFETY: every bit pattern of a `ppcf128` is a valid `[f64; 2]`.
        let [large, small] = unsafe { crate::mem::transmute::<ppcf128, [f64; 2]>(self) };
        (large, small)
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
