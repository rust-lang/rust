//! # References:
//!
//! - Section 8.3 "16-bit multiplications"
//!
//! Intrinsics that could live here:
//!
//! - [ ] __smulbb
//! - [ ] __smulbt
//! - [ ] __smultb
//! - [ ] __smultt
//! - [ ] __smulwb
//! - [ ] __smulwt
//! - [x] __qadd
//! - [x] __qsub
//! - [ ] __qdbl
//! - [ ] __smlabb
//! - [ ] __smlabt
//! - [ ] __smlatb
//! - [ ] __smlatt
//! - [ ] __smlawb
//! - [ ] __smlawt

extern "C" {
    #[link_name = "llvm.arm.qadd"]
    fn arm_qadd(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.qsub"]
    fn arm_qsub(a: i32, b: i32) -> i32;

}

/// Signed saturating addition
///
/// Returns the 32-bit saturating signed equivalent of a + b.
#[inline]
#[cfg_attr(test, assert_instr(qadd))]
pub unsafe fn qadd(a: i32, b: i32) -> i32 {
    arm_qadd(a, b)
}

/// Signed saturating subtraction
///
/// Returns the 32-bit saturating signed equivalent of a - b.
#[inline]
#[cfg_attr(test, assert_instr(qsub))]
pub unsafe fn qsub(a: i32, b: i32) -> i32 {
    arm_qsub(a, b)
}
