//! MIPS

mod msa;
pub use self::msa::*;

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Generates the trap instruction `BREAK`
#[cfg_attr(test, assert_instr(break))]
#[inline]
pub unsafe fn break_() -> ! {
    crate::intrinsics::abort()
}
