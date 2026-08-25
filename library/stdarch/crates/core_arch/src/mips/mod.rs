//! MIPS

// Building this module (even if unused) for non-fp64 targets fails with an LLVM
// error.
#[cfg(target_feature = "fp64")]
mod msa;
#[cfg(target_feature = "fp64")]
#[unstable(feature = "stdarch_mips", issue = "111198")]
pub use self::msa::*;

#[cfg(test)]
use stdarch_test::assert_instr;

/// Generates the trap instruction `BREAK`
#[cfg_attr(test, assert_instr(break))]
#[inline]
#[unstable(feature = "stdarch_mips", issue = "111198")]
pub unsafe fn break_() -> ! {
    crate::intrinsics::abort()
}
