//! MIPS

// Building this module (even if unused) for non-fp64 targets such as the Sony
// PSP fails with an LLVM error. There doesn't seem to be a good way to detect
// fp64 support as it is sometimes implied by the target cpu, so
// `#[cfg(target_feature = "fp64")]` will unfortunately not work. This is a
// fairly conservative workaround that only disables MSA intrinsics for the PSP.
#[cfg(not(target_os = "psp"))]
mod msa;
#[cfg(not(target_os = "psp"))]
pub use self::msa::*;

#[cfg(test)]
use stdarch_test::assert_instr;

/// Generates the trap instruction `BREAK`
#[cfg_attr(test, assert_instr(break))]
#[inline]
pub unsafe fn break_() -> ! {
    crate::intrinsics::abort()
}
