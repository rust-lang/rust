//! PowerPC intrinsics

#[macro_use]
mod macros;

#[cfg(any(target_feature = "altivec", doc))]
mod altivec;
#[cfg(any(target_feature = "altivec", doc))]
pub use self::altivec::*;

#[cfg(any(target_feature = "vsx", doc))]
mod vsx;
#[cfg(any(target_feature = "vsx", doc))]
pub use self::vsx::*;

#[cfg(test)]
use stdarch_test::assert_instr;

/// Generates the trap instruction `TRAP`
#[cfg_attr(test, assert_instr(trap))]
#[inline]
pub unsafe fn trap() -> ! {
    crate::intrinsics::abort()
}
