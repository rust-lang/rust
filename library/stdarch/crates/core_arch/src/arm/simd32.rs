//! # References
//!
//! - Section 8.5 "32-bit SIMD intrinsics" of ACLE
//!
//! Intrinsics that could live here
//!
//! - \[x\] __sel
//! - \[ \] __ssat16
//! - \[ \] __usat16
//! - \[ \] __sxtab16
//! - \[ \] __sxtb16
//! - \[ \] __uxtab16
//! - \[ \] __uxtb16
//! - \[x\] __qadd8
//! - \[x\] __qsub8
//! - \[x\] __sadd8
//! - \[x\] __shadd8
//! - \[x\] __shsub8
//! - \[x\] __ssub8
//! - \[ \] __uadd8
//! - \[ \] __uhadd8
//! - \[ \] __uhsub8
//! - \[ \] __uqadd8
//! - \[ \] __uqsub8
//! - \[x\] __usub8
//! - \[x\] __usad8
//! - \[x\] __usada8
//! - \[x\] __qadd16
//! - \[x\] __qasx
//! - \[x\] __qsax
//! - \[x\] __qsub16
//! - \[x\] __sadd16
//! - \[x\] __sasx
//! - \[x\] __shadd16
//! - \[ \] __shasx
//! - \[ \] __shsax
//! - \[x\] __shsub16
//! - \[ \] __ssax
//! - \[ \] __ssub16
//! - \[ \] __uadd16
//! - \[ \] __uasx
//! - \[ \] __uhadd16
//! - \[ \] __uhasx
//! - \[ \] __uhsax
//! - \[ \] __uhsub16
//! - \[ \] __uqadd16
//! - \[ \] __uqasx
//! - \[x\] __uqsax
//! - \[ \] __uqsub16
//! - \[ \] __usax
//! - \[ \] __usub16
//! - \[x\] __smlad
//! - \[ \] __smladx
//! - \[ \] __smlald
//! - \[ \] __smlaldx
//! - \[x\] __smlsd
//! - \[ \] __smlsdx
//! - \[ \] __smlsld
//! - \[ \] __smlsldx
//! - \[x\] __smuad
//! - \[x\] __smuadx
//! - \[x\] __smusd
//! - \[x\] __smusdx

#[cfg(test)]
use stdarch_test::assert_instr;

use crate::mem::transmute;

/// ARM-specific vector of four packed `i8` packed into a 32-bit integer.
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub type int8x4_t = i32;

/// ARM-specific vector of four packed `u8` packed into a 32-bit integer.
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub type uint8x4_t = u32;

/// ARM-specific vector of two packed `i16` packed into a 32-bit integer.
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub type int16x2_t = i32;

/// ARM-specific vector of two packed `u16` packed into a 32-bit integer.
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub type uint16x2_t = u32;

macro_rules! dsp_call {
    ($name:expr, $a:expr, $b:expr) => {
        transmute($name(transmute($a), transmute($b)))
    };
}

unsafe extern "unadjusted" {
    #[link_name = "llvm.arm.qadd8"]
    fn arm_qadd8(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.qsub8"]
    fn arm_qsub8(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.qsub16"]
    fn arm_qsub16(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.qadd16"]
    fn arm_qadd16(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.qasx"]
    fn arm_qasx(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.qsax"]
    fn arm_qsax(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.sadd16"]
    fn arm_sadd16(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.sadd8"]
    fn arm_sadd8(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.smlad"]
    fn arm_smlad(a: i32, b: i32, c: i32) -> i32;

    #[link_name = "llvm.arm.smlsd"]
    fn arm_smlsd(a: i32, b: i32, c: i32) -> i32;

    #[link_name = "llvm.arm.sasx"]
    fn arm_sasx(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.sel"]
    fn arm_sel(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.shadd8"]
    fn arm_shadd8(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.shadd16"]
    fn arm_shadd16(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.shsub8"]
    fn arm_shsub8(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.ssub8"]
    fn arm_ssub8(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.usub8"]
    fn arm_usub8(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.shsub16"]
    fn arm_shsub16(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.smuad"]
    fn arm_smuad(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.smuadx"]
    fn arm_smuadx(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.smusd"]
    fn arm_smusd(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.smusdx"]
    fn arm_smusdx(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.usad8"]
    fn arm_usad8(a: i32, b: i32) -> u32;
}

/// Saturating four 8-bit integer additions
///
/// Returns the 8-bit signed equivalent of
///
/// res\[0\] = a\[0\] + b\[0\]
/// res\[1\] = a\[1\] + b\[1\]
/// res\[2\] = a\[2\] + b\[2\]
/// res\[3\] = a\[3\] + b\[3\]
#[inline]
#[cfg_attr(test, assert_instr(qadd8))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __qadd8(a: int8x4_t, b: int8x4_t) -> int8x4_t {
    dsp_call!(arm_qadd8, a, b)
}

/// Saturating two 8-bit integer subtraction
///
/// Returns the 8-bit signed equivalent of
///
/// res\[0\] = a\[0\] - b\[0\]
/// res\[1\] = a\[1\] - b\[1\]
/// res\[2\] = a\[2\] - b\[2\]
/// res\[3\] = a\[3\] - b\[3\]
#[inline]
#[cfg_attr(test, assert_instr(qsub8))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __qsub8(a: int8x4_t, b: int8x4_t) -> int8x4_t {
    dsp_call!(arm_qsub8, a, b)
}

/// Saturating two 16-bit integer subtraction
///
/// Returns the 16-bit signed equivalent of
///
/// res\[0\] = a\[0\] - b\[0\]
/// res\[1\] = a\[1\] - b\[1\]
#[inline]
#[cfg_attr(test, assert_instr(qsub16))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __qsub16(a: int16x2_t, b: int16x2_t) -> int16x2_t {
    dsp_call!(arm_qsub16, a, b)
}

/// Saturating two 16-bit integer additions
///
/// Returns the 16-bit signed equivalent of
///
/// res\[0\] = a\[0\] + b\[0\]
/// res\[1\] = a\[1\] + b\[1\]
#[inline]
#[cfg_attr(test, assert_instr(qadd16))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __qadd16(a: int16x2_t, b: int16x2_t) -> int16x2_t {
    dsp_call!(arm_qadd16, a, b)
}

/// Returns the 16-bit signed saturated equivalent of
///
/// res\[0\] = a\[0\] - b\[1\]
/// res\[1\] = a\[1\] + b\[0\]
#[inline]
#[cfg_attr(test, assert_instr(qasx))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __qasx(a: int16x2_t, b: int16x2_t) -> int16x2_t {
    dsp_call!(arm_qasx, a, b)
}

/// Returns the 16-bit signed saturated equivalent of
///
/// res\[0\] = a\[0\] + b\[1\]
/// res\[1\] = a\[1\] - b\[0\]
#[inline]
#[cfg_attr(test, assert_instr(qsax))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __qsax(a: int16x2_t, b: int16x2_t) -> int16x2_t {
    dsp_call!(arm_qsax, a, b)
}

/// Returns the 16-bit signed saturated equivalent of
///
/// res\[0\] = a\[0\] + b\[1\]
/// res\[1\] = a\[1\] + b\[0\]
///
/// and the GE bits of the APSR are set.
#[inline]
#[cfg_attr(test, assert_instr(sadd16))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __sadd16(a: int16x2_t, b: int16x2_t) -> int16x2_t {
    dsp_call!(arm_sadd16, a, b)
}

/// Returns the 8-bit signed saturated equivalent of
///
/// res\[0\] = a\[0\] + b\[1\]
/// res\[1\] = a\[1\] + b\[0\]
/// res\[2\] = a\[2\] + b\[2\]
/// res\[3\] = a\[3\] + b\[3\]
///
/// and the GE bits of the APSR are set.
#[inline]
#[cfg_attr(test, assert_instr(sadd8))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __sadd8(a: int8x4_t, b: int8x4_t) -> int8x4_t {
    dsp_call!(arm_sadd8, a, b)
}

/// Dual 16-bit Signed Multiply with Addition of products
/// and 32-bit accumulation.
///
/// Returns the 16-bit signed equivalent of
/// res = a\[0\] * b\[0\] + a\[1\] * b\[1\] + c
#[inline]
#[cfg_attr(test, assert_instr(smlad))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __smlad(a: int16x2_t, b: int16x2_t, c: i32) -> i32 {
    arm_smlad(transmute(a), transmute(b), c)
}

/// Dual 16-bit Signed Multiply with Subtraction  of products
/// and 32-bit accumulation and overflow detection.
///
/// Returns the 16-bit signed equivalent of
/// res = a\[0\] * b\[0\] - a\[1\] * b\[1\] + c
#[inline]
#[cfg_attr(test, assert_instr(smlsd))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __smlsd(a: int16x2_t, b: int16x2_t, c: i32) -> i32 {
    arm_smlsd(transmute(a), transmute(b), c)
}

/// Returns the 16-bit signed equivalent of
///
/// res\[0\] = a\[0\] - b\[1\]
/// res\[1\] = a\[1\] + b\[0\]
///
/// and the GE bits of the APSR are set.
#[inline]
#[cfg_attr(test, assert_instr(sasx))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __sasx(a: int16x2_t, b: int16x2_t) -> int16x2_t {
    dsp_call!(arm_sasx, a, b)
}

/// Select bytes from each operand according to APSR GE flags
///
/// Returns the equivalent of
///
/// res\[0\] = GE\[0\] ? a\[0\] : b\[0\]
/// res\[1\] = GE\[1\] ? a\[1\] : b\[1\]
/// res\[2\] = GE\[2\] ? a\[2\] : b\[2\]
/// res\[3\] = GE\[3\] ? a\[3\] : b\[3\]
///
/// where GE are bits of APSR
#[inline]
#[cfg_attr(test, assert_instr(sel))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __sel(a: int8x4_t, b: int8x4_t) -> int8x4_t {
    dsp_call!(arm_sel, a, b)
}

/// Signed halving parallel byte-wise addition.
///
/// Returns the 8-bit signed equivalent of
///
/// res\[0\] = (a\[0\] + b\[0\]) / 2
/// res\[1\] = (a\[1\] + b\[1\]) / 2
/// res\[2\] = (a\[2\] + b\[2\]) / 2
/// res\[3\] = (a\[3\] + b\[3\]) / 2
#[inline]
#[cfg_attr(test, assert_instr(shadd8))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __shadd8(a: int8x4_t, b: int8x4_t) -> int8x4_t {
    dsp_call!(arm_shadd8, a, b)
}

/// Signed halving parallel halfword-wise addition.
///
/// Returns the 16-bit signed equivalent of
///
/// res\[0\] = (a\[0\] + b\[0\]) / 2
/// res\[1\] = (a\[1\] + b\[1\]) / 2
#[inline]
#[cfg_attr(test, assert_instr(shadd16))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __shadd16(a: int16x2_t, b: int16x2_t) -> int16x2_t {
    dsp_call!(arm_shadd16, a, b)
}

/// Signed halving parallel byte-wise subtraction.
///
/// Returns the 8-bit signed equivalent of
///
/// res\[0\] = (a\[0\] - b\[0\]) / 2
/// res\[1\] = (a\[1\] - b\[1\]) / 2
/// res\[2\] = (a\[2\] - b\[2\]) / 2
/// res\[3\] = (a\[3\] - b\[3\]) / 2
#[inline]
#[cfg_attr(test, assert_instr(shsub8))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __shsub8(a: int8x4_t, b: int8x4_t) -> int8x4_t {
    dsp_call!(arm_shsub8, a, b)
}

/// Inserts a `USUB8` instruction.
///
/// Returns the 8-bit unsigned equivalent of
///
/// res\[0\] = a\[0\] - a\[0\]
/// res\[1\] = a\[1\] - a\[1\]
/// res\[2\] = a\[2\] - a\[2\]
/// res\[3\] = a\[3\] - a\[3\]
///
/// where \[0\] is the lower 8 bits and \[3\] is the upper 8 bits.
/// The GE bits of the APSR are set.
#[inline]
#[cfg_attr(test, assert_instr(usub8))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __usub8(a: uint8x4_t, b: uint8x4_t) -> uint8x4_t {
    dsp_call!(arm_usub8, a, b)
}

/// Inserts a `SSUB8` instruction.
///
/// Returns the 8-bit signed equivalent of
///
/// res\[0\] = a\[0\] - a\[0\]
/// res\[1\] = a\[1\] - a\[1\]
/// res\[2\] = a\[2\] - a\[2\]
/// res\[3\] = a\[3\] - a\[3\]
///
/// where \[0\] is the lower 8 bits and \[3\] is the upper 8 bits.
/// The GE bits of the APSR are set.
#[inline]
#[cfg_attr(test, assert_instr(ssub8))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __ssub8(a: int8x4_t, b: int8x4_t) -> int8x4_t {
    dsp_call!(arm_ssub8, a, b)
}

/// Signed halving parallel halfword-wise subtraction.
///
/// Returns the 16-bit signed equivalent of
///
/// res\[0\] = (a\[0\] - b\[0\]) / 2
/// res\[1\] = (a\[1\] - b\[1\]) / 2
#[inline]
#[cfg_attr(test, assert_instr(shsub16))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __shsub16(a: int16x2_t, b: int16x2_t) -> int16x2_t {
    dsp_call!(arm_shsub16, a, b)
}

/// Signed Dual Multiply Add.
///
/// Returns the equivalent of
///
/// res = a\[0\] * b\[0\] + a\[1\] * b\[1\]
///
/// and sets the Q flag if overflow occurs on the addition.
#[inline]
#[cfg_attr(test, assert_instr(smuad))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __smuad(a: int16x2_t, b: int16x2_t) -> i32 {
    arm_smuad(transmute(a), transmute(b))
}

/// Signed Dual Multiply Add Reversed.
///
/// Returns the equivalent of
///
/// res = a\[0\] * b\[1\] + a\[1\] * b\[0\]
///
/// and sets the Q flag if overflow occurs on the addition.
#[inline]
#[cfg_attr(test, assert_instr(smuadx))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __smuadx(a: int16x2_t, b: int16x2_t) -> i32 {
    arm_smuadx(transmute(a), transmute(b))
}

/// Signed Dual Multiply Subtract.
///
/// Returns the equivalent of
///
/// res = a\[0\] * b\[0\] - a\[1\] * b\[1\]
///
/// and sets the Q flag if overflow occurs on the addition.
#[inline]
#[cfg_attr(test, assert_instr(smusd))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __smusd(a: int16x2_t, b: int16x2_t) -> i32 {
    arm_smusd(transmute(a), transmute(b))
}

/// Signed Dual Multiply Subtract Reversed.
///
/// Returns the equivalent of
///
/// res = a\[0\] * b\[1\] - a\[1\] * b\[0\]
///
/// and sets the Q flag if overflow occurs on the addition.
#[inline]
#[cfg_attr(test, assert_instr(smusdx))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __smusdx(a: int16x2_t, b: int16x2_t) -> i32 {
    arm_smusdx(transmute(a), transmute(b))
}

/// Sum of 8-bit absolute differences.
///
/// Returns the 8-bit unsigned equivalent of
///
/// res = abs(a\[0\] - b\[0\]) + abs(a\[1\] - b\[1\]) +\
///          (a\[2\] - b\[2\]) + (a\[3\] - b\[3\])
#[inline]
#[cfg_attr(test, assert_instr(usad8))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __usad8(a: int8x4_t, b: int8x4_t) -> u32 {
    arm_usad8(transmute(a), transmute(b))
}

/// Sum of 8-bit absolute differences and constant.
///
/// Returns the 8-bit unsigned equivalent of
///
/// res = abs(a\[0\] - b\[0\]) + abs(a\[1\] - b\[1\]) +\
///          (a\[2\] - b\[2\]) + (a\[3\] - b\[3\]) + c
#[inline]
#[cfg_attr(test, assert_instr(usad8))]
#[unstable(feature = "stdarch_arm_dsp", issue = "117237")]
pub unsafe fn __usada8(a: int8x4_t, b: int8x4_t, c: u32) -> u32 {
    __usad8(a, b) + c
}

#[cfg(test)]
mod tests {
    use crate::core_arch::simd::{i8x4, i16x2, u8x4};
    use std::mem::transmute;
    use stdarch_test::simd_test;

    #[test]
    fn qadd8() {
        unsafe {
            let a = i8x4::new(1, 2, 3, i8::MAX);
            let b = i8x4::new(2, -1, 0, 1);
            let c = i8x4::new(3, 1, 3, i8::MAX);
            let r: i8x4 = dsp_call!(super::__qadd8, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn qsub8() {
        unsafe {
            let a = i8x4::new(1, 2, 3, i8::MIN);
            let b = i8x4::new(2, -1, 0, 1);
            let c = i8x4::new(-1, 3, 3, i8::MIN);
            let r: i8x4 = dsp_call!(super::__qsub8, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn qadd16() {
        unsafe {
            let a = i16x2::new(1, 2);
            let b = i16x2::new(2, -1);
            let c = i16x2::new(3, 1);
            let r: i16x2 = dsp_call!(super::__qadd16, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn qsub16() {
        unsafe {
            let a = i16x2::new(10, 20);
            let b = i16x2::new(20, -10);
            let c = i16x2::new(-10, 30);
            let r: i16x2 = dsp_call!(super::__qsub16, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn qasx() {
        unsafe {
            let a = i16x2::new(1, i16::MAX);
            let b = i16x2::new(2, 2);
            let c = i16x2::new(-1, i16::MAX);
            let r: i16x2 = dsp_call!(super::__qasx, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn qsax() {
        unsafe {
            let a = i16x2::new(1, i16::MAX);
            let b = i16x2::new(2, 2);
            let c = i16x2::new(3, i16::MAX - 2);
            let r: i16x2 = dsp_call!(super::__qsax, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn sadd16() {
        unsafe {
            let a = i16x2::new(1, i16::MAX);
            let b = i16x2::new(2, 2);
            let c = i16x2::new(3, -i16::MAX);
            let r: i16x2 = dsp_call!(super::__sadd16, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn sadd8() {
        unsafe {
            let a = i8x4::new(1, 2, 3, i8::MAX);
            let b = i8x4::new(4, 3, 2, 2);
            let c = i8x4::new(5, 5, 5, -i8::MAX);
            let r: i8x4 = dsp_call!(super::__sadd8, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn sasx() {
        unsafe {
            let a = i16x2::new(1, 2);
            let b = i16x2::new(2, 1);
            let c = i16x2::new(0, 4);
            let r: i16x2 = dsp_call!(super::__sasx, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn smlad() {
        unsafe {
            let a = i16x2::new(1, 2);
            let b = i16x2::new(3, 4);
            let r = super::__smlad(transmute(a), transmute(b), 10);
            assert_eq!(r, (1 * 3) + (2 * 4) + 10);
        }
    }

    #[test]
    fn smlsd() {
        unsafe {
            let a = i16x2::new(1, 2);
            let b = i16x2::new(3, 4);
            let r = super::__smlsd(transmute(a), transmute(b), 10);
            assert_eq!(r, ((1 * 3) - (2 * 4)) + 10);
        }
    }

    #[test]
    fn sel() {
        unsafe {
            let a = i8x4::new(1, 2, 3, i8::MAX);
            let b = i8x4::new(4, 3, 2, 2);
            // call sadd8() to set GE bits
            super::__sadd8(transmute(a), transmute(b));
            let c = i8x4::new(1, 2, 3, i8::MAX);
            let r: i8x4 = dsp_call!(super::__sel, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn shadd8() {
        unsafe {
            let a = i8x4::new(1, 2, 3, 4);
            let b = i8x4::new(5, 4, 3, 2);
            let c = i8x4::new(3, 3, 3, 3);
            let r: i8x4 = dsp_call!(super::__shadd8, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn shadd16() {
        unsafe {
            let a = i16x2::new(1, 2);
            let b = i16x2::new(5, 4);
            let c = i16x2::new(3, 3);
            let r: i16x2 = dsp_call!(super::__shadd16, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn shsub8() {
        unsafe {
            let a = i8x4::new(1, 2, 3, 4);
            let b = i8x4::new(5, 4, 3, 2);
            let c = i8x4::new(-2, -1, 0, 1);
            let r: i8x4 = dsp_call!(super::__shsub8, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn ssub8() {
        unsafe {
            let a = i8x4::new(1, 2, 3, 4);
            let b = i8x4::new(5, 4, 3, 2);
            let c = i8x4::new(-4, -2, 0, 2);
            let r: i8x4 = dsp_call!(super::__ssub8, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn usub8() {
        unsafe {
            let a = u8x4::new(1, 2, 3, 4);
            let b = u8x4::new(5, 4, 3, 2);
            let c = u8x4::new(252, 254, 0, 2);
            let r: u8x4 = dsp_call!(super::__usub8, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn shsub16() {
        unsafe {
            let a = i16x2::new(1, 2);
            let b = i16x2::new(5, 4);
            let c = i16x2::new(-2, -1);
            let r: i16x2 = dsp_call!(super::__shsub16, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn smuad() {
        unsafe {
            let a = i16x2::new(1, 2);
            let b = i16x2::new(5, 4);
            let r = super::__smuad(transmute(a), transmute(b));
            assert_eq!(r, 13);
        }
    }

    #[test]
    fn smuadx() {
        unsafe {
            let a = i16x2::new(1, 2);
            let b = i16x2::new(5, 4);
            let r = super::__smuadx(transmute(a), transmute(b));
            assert_eq!(r, 14);
        }
    }

    #[test]
    fn smusd() {
        unsafe {
            let a = i16x2::new(1, 2);
            let b = i16x2::new(5, 4);
            let r = super::__smusd(transmute(a), transmute(b));
            assert_eq!(r, -3);
        }
    }

    #[test]
    fn smusdx() {
        unsafe {
            let a = i16x2::new(1, 2);
            let b = i16x2::new(5, 4);
            let r = super::__smusdx(transmute(a), transmute(b));
            assert_eq!(r, -6);
        }
    }

    #[test]
    fn usad8() {
        unsafe {
            let a = i8x4::new(1, 2, 3, 4);
            let b = i8x4::new(4, 3, 2, 1);
            let r = super::__usad8(transmute(a), transmute(b));
            assert_eq!(r, 8);
        }
    }

    #[test]
    fn usad8a() {
        unsafe {
            let a = i8x4::new(1, 2, 3, 4);
            let b = i8x4::new(4, 3, 2, 1);
            let c = 10;
            let r = super::__usada8(transmute(a), transmute(b), c);
            assert_eq!(r, 8 + c);
        }
    }
}
