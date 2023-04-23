//! PowerPC intrinsics

#[macro_use]
mod macros;

mod altivec;
pub use self::altivec::*;

mod vsx;
pub use self::vsx::*;

#[cfg(test)]
use stdarch_test::assert_instr;

/// Generates the trap instruction `TRAP`
#[cfg_attr(test, assert_instr(trap))]
#[inline]
pub unsafe fn trap() -> ! {
    crate::intrinsics::abort()
}
