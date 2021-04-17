//! # References:
//!
//! - Section 8.3 "16-bit multiplications"
//!
//! Intrinsics that could live here:
//!
//! - \[x\] __smulbb
//! - \[x\] __smulbt
//! - \[x\] __smultb
//! - \[x\] __smultt
//! - \[x\] __smulwb
//! - \[x\] __smulwt
//! - \[x\] __qadd
//! - \[x\] __qsub
//! - \[x\] __qdbl
//! - \[x\] __smlabb
//! - \[x\] __smlabt
//! - \[x\] __smlatb
//! - \[x\] __smlatt
//! - \[x\] __smlawb
//! - \[x\] __smlawt

#[cfg(test)]
use stdarch_test::assert_instr;

use crate::mem::transmute;

types! {
    /// ARM-specific 32-bit wide vector of two packed `i16`.
    pub struct int16x2_t(i16, i16);
    /// ARM-specific 32-bit wide vector of two packed `u16`.
    pub struct uint16x2_t(u16, u16);
}

extern "C" {
    #[link_name = "llvm.arm.smulbb"]
    fn arm_smulbb(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.smulbt"]
    fn arm_smulbt(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.smultb"]
    fn arm_smultb(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.smultt"]
    fn arm_smultt(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.smulwb"]
    fn arm_smulwb(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.smulwt"]
    fn arm_smulwt(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.qadd"]
    fn arm_qadd(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.qsub"]
    fn arm_qsub(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.smlabb"]
    fn arm_smlabb(a: i32, b: i32, c: i32) -> i32;

    #[link_name = "llvm.arm.smlabt"]
    fn arm_smlabt(a: i32, b: i32, c: i32) -> i32;

    #[link_name = "llvm.arm.smlatb"]
    fn arm_smlatb(a: i32, b: i32, c: i32) -> i32;

    #[link_name = "llvm.arm.smlatt"]
    fn arm_smlatt(a: i32, b: i32, c: i32) -> i32;

    #[link_name = "llvm.arm.smlawb"]
    fn arm_smlawb(a: i32, b: i32, c: i32) -> i32;

    #[link_name = "llvm.arm.smlawt"]
    fn arm_smlawt(a: i32, b: i32, c: i32) -> i32;
}

/// Insert a SMULBB instruction
///
/// Returns the equivalent of a\[0\] * b\[0\]
/// where \[0\] is the lower 16 bits and \[1\] is the upper 16 bits.
#[inline]
#[cfg_attr(test, assert_instr(smulbb))]
pub unsafe fn __smulbb(a: int16x2_t, b: int16x2_t) -> i32 {
    arm_smulbb(transmute(a), transmute(b))
}

/// Insert a SMULTB instruction
///
/// Returns the equivalent of a\[0\] * b\[1\]
/// where \[0\] is the lower 16 bits and \[1\] is the upper 16 bits.
#[inline]
#[cfg_attr(test, assert_instr(smultb))]
pub unsafe fn __smultb(a: int16x2_t, b: int16x2_t) -> i32 {
    arm_smultb(transmute(a), transmute(b))
}

/// Insert a SMULTB instruction
///
/// Returns the equivalent of a\[1\] * b\[0\]
/// where \[0\] is the lower 16 bits and \[1\] is the upper 16 bits.
#[inline]
#[cfg_attr(test, assert_instr(smulbt))]
pub unsafe fn __smulbt(a: int16x2_t, b: int16x2_t) -> i32 {
    arm_smulbt(transmute(a), transmute(b))
}

/// Insert a SMULTT instruction
///
/// Returns the equivalent of a\[1\] * b\[1\]
/// where \[0\] is the lower 16 bits and \[1\] is the upper 16 bits.
#[inline]
#[cfg_attr(test, assert_instr(smultt))]
pub unsafe fn __smultt(a: int16x2_t, b: int16x2_t) -> i32 {
    arm_smultt(transmute(a), transmute(b))
}

/// Insert a SMULWB instruction
///
/// Multiplies the 32-bit signed first operand with the low halfword
/// (as a 16-bit signed integer) of the second operand.
/// Return the top 32 bits of the 48-bit product
#[inline]
#[cfg_attr(test, assert_instr(smulwb))]
pub unsafe fn __smulwb(a: int16x2_t, b: i32) -> i32 {
    arm_smulwb(transmute(a), b)
}

/// Insert a SMULWT instruction
///
/// Multiplies the 32-bit signed first operand with the high halfword
/// (as a 16-bit signed integer) of the second operand.
/// Return the top 32 bits of the 48-bit product
#[inline]
#[cfg_attr(test, assert_instr(smulwt))]
pub unsafe fn __smulwt(a: int16x2_t, b: i32) -> i32 {
    arm_smulwt(transmute(a), b)
}

/// Signed saturating addition
///
/// Returns the 32-bit saturating signed equivalent of a + b.
/// Sets the Q flag if saturation occurs.
#[inline]
#[cfg_attr(test, assert_instr(qadd))]
pub unsafe fn __qadd(a: i32, b: i32) -> i32 {
    arm_qadd(a, b)
}

/// Signed saturating subtraction
///
/// Returns the 32-bit saturating signed equivalent of a - b.
/// Sets the Q flag if saturation occurs.
#[inline]
#[cfg_attr(test, assert_instr(qsub))]
pub unsafe fn __qsub(a: i32, b: i32) -> i32 {
    arm_qsub(a, b)
}

/// Insert a QADD instruction
///
/// Returns the 32-bit saturating signed equivalent of a + a
/// Sets the Q flag if saturation occurs.
#[inline]
#[cfg_attr(test, assert_instr(qadd))]
pub unsafe fn __qdbl(a: i32) -> i32 {
    arm_qadd(a, a)
}

/// Insert a SMLABB instruction
///
/// Returns the equivalent of a\[0\] * b\[0\] + c
/// where \[0\] is the lower 16 bits and \[1\] is the upper 16 bits.
/// Sets the Q flag if overflow occurs on the addition.
#[inline]
#[cfg_attr(test, assert_instr(smlabb))]
pub unsafe fn __smlabb(a: int16x2_t, b: int16x2_t, c: i32) -> i32 {
    arm_smlabb(transmute(a), transmute(b), c)
}

/// Insert a SMLABT instruction
///
/// Returns the equivalent of a\[0\] * b\[1\] + c
/// where \[0\] is the lower 16 bits and \[1\] is the upper 16 bits.
/// Sets the Q flag if overflow occurs on the addition.
#[inline]
#[cfg_attr(test, assert_instr(smlabt))]
pub unsafe fn __smlabt(a: int16x2_t, b: int16x2_t, c: i32) -> i32 {
    arm_smlabt(transmute(a), transmute(b), c)
}

/// Insert a SMLATB instruction
///
/// Returns the equivalent of a\[1\] * b\[0\] + c
/// where \[0\] is the lower 16 bits and \[1\] is the upper 16 bits.
/// Sets the Q flag if overflow occurs on the addition.
#[inline]
#[cfg_attr(test, assert_instr(smlatb))]
pub unsafe fn __smlatb(a: int16x2_t, b: int16x2_t, c: i32) -> i32 {
    arm_smlatb(transmute(a), transmute(b), c)
}

/// Insert a SMLATT instruction
///
/// Returns the equivalent of a\[1\] * b\[1\] + c
/// where \[0\] is the lower 16 bits and \[1\] is the upper 16 bits.
/// Sets the Q flag if overflow occurs on the addition.
#[inline]
#[cfg_attr(test, assert_instr(smlatt))]
pub unsafe fn __smlatt(a: int16x2_t, b: int16x2_t, c: i32) -> i32 {
    arm_smlatt(transmute(a), transmute(b), c)
}

/// Insert a SMLAWB instruction
///
/// Returns the equivalent of (a * b\[0\] + (c << 16)) >> 16
/// where \[0\] is the lower 16 bits and \[1\] is the upper 16 bits.
/// Sets the Q flag if overflow occurs on the addition.
#[inline]
#[cfg_attr(test, assert_instr(smlawb))]
pub unsafe fn __smlawb(a: i32, b: int16x2_t, c: i32) -> i32 {
    arm_smlawb(a, transmute(b), c)
}

/// Insert a SMLAWT instruction
///
/// Returns the equivalent of (a * b\[1\] + (c << 16)) >> 16
/// where \[0\] is the lower 16 bits and \[1\] is the upper 16 bits.
/// Sets the Q flag if overflow occurs on the addition.
#[inline]
#[cfg_attr(test, assert_instr(smlawt))]
pub unsafe fn __smlawt(a: i32, b: int16x2_t, c: i32) -> i32 {
    arm_smlawt(a, transmute(b), c)
}

#[cfg(test)]
mod tests {
    use crate::core_arch::{
        arm::*,
        simd::{i16x2, i8x4, u8x4},
    };
    use std::mem::transmute;
    use stdarch_test::simd_test;

    #[test]
    fn smulbb() {
        unsafe {
            let a = i16x2::new(10, 20);
            let b = i16x2::new(30, 40);
            assert_eq!(super::__smulbb(transmute(a), transmute(b)), 10 * 30);
        }
    }

    #[test]
    fn smulbt() {
        unsafe {
            let a = i16x2::new(10, 20);
            let b = i16x2::new(30, 40);
            assert_eq!(super::__smulbt(transmute(a), transmute(b)), 10 * 40);
        }
    }

    #[test]
    fn smultb() {
        unsafe {
            let a = i16x2::new(10, 20);
            let b = i16x2::new(30, 40);
            assert_eq!(super::__smultb(transmute(a), transmute(b)), 20 * 30);
        }
    }

    #[test]
    fn smultt() {
        unsafe {
            let a = i16x2::new(10, 20);
            let b = i16x2::new(30, 40);
            assert_eq!(super::__smultt(transmute(a), transmute(b)), 20 * 40);
        }
    }

    #[test]
    fn smulwb() {
        unsafe {
            let a = i16x2::new(10, 20);
            let b = 30;
            assert_eq!(super::__smulwb(transmute(a), b), 20 * b);
        }
    }

    #[test]
    fn smulwt() {
        unsafe {
            let a = i16x2::new(10, 20);
            let b = 30;
            assert_eq!(super::__smulwt(transmute(a), b), (10 * b) >> 16);
        }
    }

    #[test]
    fn qadd() {
        unsafe {
            assert_eq!(super::__qadd(-10, 60), 50);
            assert_eq!(super::__qadd(i32::MAX, 10), i32::MAX);
            assert_eq!(super::__qadd(i32::MIN, -10), i32::MIN);
        }
    }

    #[test]
    fn qsub() {
        unsafe {
            assert_eq!(super::__qsub(10, 60), -50);
            assert_eq!(super::__qsub(i32::MAX, -10), i32::MAX);
            assert_eq!(super::__qsub(i32::MIN, 10), i32::MIN);
        }
    }

    fn qdbl() {
        unsafe {
            assert_eq!(super::__qdbl(10), 20);
            assert_eq!(super::__qdbl(i32::MAX), i32::MAX);
        }
    }

    fn smlabb() {
        unsafe {
            let a = i16x2::new(10, 20);
            let b = i16x2::new(30, 40);
            let c = 50;
            let r = (10 * 30) + c;
            assert_eq!(super::__smlabb(transmute(a), transmute(b), c), r);
        }
    }

    fn smlabt() {
        unsafe {
            let a = i16x2::new(10, 20);
            let b = i16x2::new(30, 40);
            let c = 50;
            let r = (10 * 40) + c;
            assert_eq!(super::__smlabt(transmute(a), transmute(b), c), r);
        }
    }

    fn smlatb() {
        unsafe {
            let a = i16x2::new(10, 20);
            let b = i16x2::new(30, 40);
            let c = 50;
            let r = (20 * 30) + c;
            assert_eq!(super::__smlabt(transmute(a), transmute(b), c), r);
        }
    }

    fn smlatt() {
        unsafe {
            let a = i16x2::new(10, 20);
            let b = i16x2::new(30, 40);
            let c = 50;
            let r = (20 * 40) + c;
            assert_eq!(super::__smlatt(transmute(a), transmute(b), c), r);
        }
    }

    fn smlawb() {
        unsafe {
            let a: i32 = 10;
            let b = i16x2::new(30, 40);
            let c: i32 = 50;
            let r: i32 = ((a * 30) + (c << 16)) >> 16;
            assert_eq!(super::__smlawb(a, transmute(b), c), r);
        }
    }

    fn smlawt() {
        unsafe {
            let a: i32 = 10;
            let b = i16x2::new(30, 40);
            let c: i32 = 50;
            let r: i32 = ((a * 40) + (c << 16)) >> 16;
            assert_eq!(super::__smlawt(a, transmute(b), c), r);
        }
    }
}
