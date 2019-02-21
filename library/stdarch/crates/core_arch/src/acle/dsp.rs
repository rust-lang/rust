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

#[cfg(test)]
use stdsimd_test::assert_instr;

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
pub unsafe fn __qadd(a: i32, b: i32) -> i32 {
    arm_qadd(a, b)
}

/// Signed saturating subtraction
///
/// Returns the 32-bit saturating signed equivalent of a - b.
#[inline]
#[cfg_attr(test, assert_instr(qsub))]
pub unsafe fn __qsub(a: i32, b: i32) -> i32 {
    arm_qsub(a, b)
}

#[cfg(test)]
mod tests {
    use crate::core_arch::arm::*;
    use std::mem;
    use stdsimd_test::simd_test;

    #[test]
    fn qadd() {
        unsafe {
            assert_eq!(super::__qadd(-10, 60), 50);
            assert_eq!(super::__qadd(::std::i32::MAX, 10), ::std::i32::MAX);
            assert_eq!(super::__qadd(::std::i32::MIN, -10), ::std::i32::MIN);
        }
    }

    #[test]
    fn qsub() {
        unsafe {
            assert_eq!(super::__qsub(10, 60), -50);
            assert_eq!(super::__qsub(::std::i32::MAX, -10), ::std::i32::MAX);
            assert_eq!(super::__qsub(::std::i32::MIN, 10), ::std::i32::MIN);
        }
    }
}
