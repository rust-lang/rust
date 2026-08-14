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
