//! PowerPC intrinsics

pub(crate) mod macros;

mod altivec;
#[unstable(feature = "stdarch_powerpc", issue = "111145")]
pub use self::altivec::*;

mod vsx;
#[unstable(feature = "stdarch_powerpc", issue = "111145")]
pub use self::vsx::*;

// When both altivec and vsx are available, prefer vsx::vec_add as is a
// is a superset of altivec::vec_add.
// This explicit re-export resolves the ambiguous_glob_reexports warning.
#[unstable(feature = "stdarch_powerpc", issue = "111145")]
pub use self::vsx::vec_add;

#[cfg(test)]
use stdarch_test::assert_instr;

/// Generates the trap instruction `TRAP`
#[cfg_attr(test, assert_instr(trap))]
#[inline]
#[unstable(feature = "stdarch_powerpc", issue = "111145")]
pub unsafe fn trap() -> ! {
    crate::intrinsics::abort()
}
