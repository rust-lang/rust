//! ARM DSP Intrinsics.

#[cfg(test)]
use stdsimd_test::assert_instr;

types! {
    /// ARM-specific 32-bit wide vector of four packed `i8`.
    pub struct int8x4_t(i8, i8, i8, i8);
    /// ARM-specific 32-bit wide vector of four packed `u8`.
    pub struct uint8x4_t(u8, u8, u8, u8);
    /// ARM-specific 32-bit wide vector of two packed `i16`.
    pub struct int16x2_t(i16, i16);
    /// ARM-specific 32-bit wide vector of two packed `u16`.
    pub struct uint16x2_t(u16, u16);
}

macro_rules! dsp_call {
    ($name:expr, $a:expr, $b:expr) => {
        ::mem::transmute($name(::mem::transmute($a), ::mem::transmute($b)))
    };
}

extern "C" {
    #[link_name = "llvm.arm.qadd"]
    fn arm_qadd(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.qadd16"]
    fn arm_qadd16(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.qadd8"]
    fn arm_qadd8(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.qasx"]
    fn arm_qasx(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.qsax"]
    fn arm_qsax(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.qsub"]
    fn arm_qsub(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.qsub8"]
    fn arm_qsub8(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.qsub16"]
    fn arm_qsub16(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.sadd16"]
    fn arm_sadd16(a: i32, b: i32) -> i32;

    #[link_name = "llvm.arm.sadd8"]
    fn arm_sadd8(a: i32, b: i32) -> i32;

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
pub unsafe fn qadd8(a: int8x4_t, b: int8x4_t) -> int8x4_t {
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
pub unsafe fn qsub8(a: int8x4_t, b: int8x4_t) -> int8x4_t {
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
pub unsafe fn qsub16(a: int16x2_t, b: int16x2_t) -> int16x2_t {
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
pub unsafe fn qadd16(a: int16x2_t, b: int16x2_t) -> int16x2_t {
    dsp_call!(arm_qadd16, a, b)
}

/// Returns the 16-bit signed saturated equivalent of
///
/// res\[0\] = a\[0\] - b\[1\]
/// res\[1\] = a\[1\] + b\[0\]
#[inline]
#[cfg_attr(test, assert_instr(qasx))]
pub unsafe fn qasx(a: int16x2_t, b: int16x2_t) -> int16x2_t {
    dsp_call!(arm_qasx, a, b)
}

/// Returns the 16-bit signed saturated equivalent of
///
/// res\[0\] = a\[0\] + b\[1\]
/// res\[1\] = a\[1\] - b\[0\]
#[inline]
#[cfg_attr(test, assert_instr(qsax))]
pub unsafe fn qsax(a: int16x2_t, b: int16x2_t) -> int16x2_t {
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
pub unsafe fn sadd16(a: int16x2_t, b: int16x2_t) -> int16x2_t {
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
pub unsafe fn sadd8(a: int8x4_t, b: int8x4_t) -> int8x4_t {
    dsp_call!(arm_sadd8, a, b)
}

/// Returns the 16-bit signed equivalent of
///
/// res\[0\] = a\[0\] - b\[1\]
/// res\[1\] = a\[1\] + b\[0\]
///
/// and the GE bits of the APSR are set.
#[inline]
#[cfg_attr(test, assert_instr(sasx))]
pub unsafe fn sasx(a: int16x2_t, b: int16x2_t) -> int16x2_t {
    dsp_call!(arm_sasx, a, b)
}

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
#[cfg(all(not(target_feature = "mclass")))]
pub unsafe fn sel(a: int8x4_t, b: int8x4_t) -> int8x4_t {
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
pub unsafe fn shadd8(a: int8x4_t, b: int8x4_t) -> int8x4_t {
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
pub unsafe fn shadd16(a: int16x2_t, b: int16x2_t) -> int16x2_t {
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
pub unsafe fn shsub8(a: int8x4_t, b: int8x4_t) -> int8x4_t {
    dsp_call!(arm_shsub8, a, b)
}

/// Signed halving parallel halfword-wise subtraction.
///
/// Returns the 16-bit signed equivalent of
///
/// res\[0\] = (a\[0\] - b\[0\]) / 2
/// res\[1\] = (a\[1\] - b\[1\]) / 2
#[inline]
#[cfg_attr(test, assert_instr(shsub16))]
pub unsafe fn shsub16(a: int16x2_t, b: int16x2_t) -> int16x2_t {
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
pub unsafe fn smuad(a: int16x2_t, b: int16x2_t) -> i32 {
    arm_smuad(::mem::transmute(a), ::mem::transmute(b))
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
pub unsafe fn smuadx(a: int16x2_t, b: int16x2_t) -> i32 {
    arm_smuadx(::mem::transmute(a), ::mem::transmute(b))
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
pub unsafe fn smusd(a: int16x2_t, b: int16x2_t) -> i32 {
    arm_smusd(::mem::transmute(a), ::mem::transmute(b))
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
pub unsafe fn smusdx(a: int16x2_t, b: int16x2_t) -> i32 {
    arm_smusdx(::mem::transmute(a), ::mem::transmute(b))
}

/// Sum of 8-bit absolute differences.
///
/// Returns the 8-bit unsigned equivalent of
///
/// res = abs(a\[0\] - b\[0\]) + abs(a\[1\] - b\[1\]) +\
///          (a\[2\] - b\[2\]) + (a\[3\] - b\[3\])
#[inline]
#[cfg_attr(test, assert_instr(usad8))]
pub unsafe fn usad8(a: int8x4_t, b: int8x4_t) -> u32 {
    arm_usad8(::mem::transmute(a), ::mem::transmute(b))
}

/// Sum of 8-bit absolute differences and constant.
///
/// Returns the 8-bit unsigned equivalent of
///
/// res = abs(a\[0\] - b\[0\]) + abs(a\[1\] - b\[1\]) +\
///          (a\[2\] - b\[2\]) + (a\[3\] - b\[3\]) + c
#[inline]
#[cfg_attr(test, assert_instr(usad8))]
pub unsafe fn usad8a(a: int8x4_t, b: int8x4_t, c: u32) -> u32 {
    usad8(a, b) + c
}

#[cfg(test)]
mod tests {
    use coresimd::arm::*;
    use coresimd::simd::*;
    use std::mem;
    use stdsimd_test::simd_test;

    #[test]
    fn qadd() {
        unsafe {
            assert_eq!(dsp::qadd(-10, 60), 50);
            assert_eq!(dsp::qadd(::std::i32::MAX, 10), ::std::i32::MAX);
            assert_eq!(dsp::qadd(::std::i32::MIN, -10), ::std::i32::MIN);
        }
    }

    #[test]
    fn qsub() {
        unsafe {
            assert_eq!(dsp::qsub(10, 60), -50);
            assert_eq!(dsp::qsub(::std::i32::MAX, -10), ::std::i32::MAX);
            assert_eq!(dsp::qsub(::std::i32::MIN, 10), ::std::i32::MIN);
        }
    }

    #[test]
    fn qadd8() {
        unsafe {
            let a = i8x4::new(1, 2, 3, ::std::i8::MAX);
            let b = i8x4::new(2, -1, 0, 1);
            let c = i8x4::new(3, 1, 3, ::std::i8::MAX);
            let r: i8x4 = dsp_call!(dsp::qadd8, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn qsub8() {
        unsafe {
            let a = i8x4::new(1, 2, 3, ::std::i8::MIN);
            let b = i8x4::new(2, -1, 0, 1);
            let c = i8x4::new(-1, 3, 3, ::std::i8::MIN);
            let r: i8x4 = dsp_call!(dsp::qsub8, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn qadd16() {
        unsafe {
            let a = i16x2::new(1, 2);
            let b = i16x2::new(2, -1);
            let c = i16x2::new(3, 1);
            let r: i16x2 = dsp_call!(dsp::qadd16, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn qsub16() {
        unsafe {
            let a = i16x2::new(10, 20);
            let b = i16x2::new(20, -10);
            let c = i16x2::new(-10, 30);
            let r: i16x2 = dsp_call!(dsp::qsub16, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn qasx() {
        unsafe {
            let a = i16x2::new(1, ::std::i16::MAX);
            let b = i16x2::new(2, 2);
            let c = i16x2::new(-1, ::std::i16::MAX);
            let r: i16x2 = dsp_call!(dsp::qasx, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn qsax() {
        unsafe {
            let a = i16x2::new(1, ::std::i16::MAX);
            let b = i16x2::new(2, 2);
            let c = i16x2::new(3, ::std::i16::MAX - 2);
            let r: i16x2 = dsp_call!(dsp::qsax, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn sadd16() {
        unsafe {
            let a = i16x2::new(1, ::std::i16::MAX);
            let b = i16x2::new(2, 2);
            let c = i16x2::new(3, -::std::i16::MAX);
            let r: i16x2 = dsp_call!(dsp::sadd16, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn sadd8() {
        unsafe {
            let a = i8x4::new(1, 2, 3, ::std::i8::MAX);
            let b = i8x4::new(4, 3, 2, 2);
            let c = i8x4::new(5, 5, 5, -::std::i8::MAX);
            let r: i8x4 = dsp_call!(dsp::sadd8, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn sasx() {
        unsafe {
            let a = i16x2::new(1, 2);
            let b = i16x2::new(2, 1);
            let c = i16x2::new(0, 4);
            let r: i16x2 = dsp_call!(dsp::sasx, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn sel() {
        unsafe {
            let a = i8x4::new(1, 2, 3, ::std::i8::MAX);
            let b = i8x4::new(4, 3, 2, 2);
            // call sadd8() to set GE bits
            dsp::sadd8(::mem::transmute(a), ::mem::transmute(b));
            let c = i8x4::new(1, 2, 3, ::std::i8::MAX);
            let r: i8x4 = dsp_call!(dsp::sel, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn shadd8() {
        unsafe {
            let a = i8x4::new(1, 2, 3, 4);
            let b = i8x4::new(5, 4, 3, 2);
            let c = i8x4::new(3, 3, 3, 3);
            let r: i8x4 = dsp_call!(dsp::shadd8, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn shadd16() {
        unsafe {
            let a = i16x2::new(1, 2);
            let b = i16x2::new(5, 4);
            let c = i16x2::new(3, 3);
            let r: i16x2 = dsp_call!(dsp::shadd16, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn shsub8() {
        unsafe {
            let a = i8x4::new(1, 2, 3, 4);
            let b = i8x4::new(5, 4, 3, 2);
            let c = i8x4::new(-2, -1, 0, 1);
            let r: i8x4 = dsp_call!(dsp::shsub8, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn shsub16() {
        unsafe {
            let a = i16x2::new(1, 2);
            let b = i16x2::new(5, 4);
            let c = i16x2::new(-2, -1);
            let r: i16x2 = dsp_call!(dsp::shsub16, a, b);
            assert_eq!(r, c);
        }
    }

    #[test]
    fn smuad() {
        unsafe {
            let a = i16x2::new(1, 2);
            let b = i16x2::new(5, 4);
            let r = dsp::smuad(::mem::transmute(a), ::mem::transmute(b));
            assert_eq!(r, 13);
        }
    }

    #[test]
    fn smuadx() {
        unsafe {
            let a = i16x2::new(1, 2);
            let b = i16x2::new(5, 4);
            let r = dsp::smuadx(::mem::transmute(a), ::mem::transmute(b));
            assert_eq!(r, 14);
        }
    }

    #[test]
    fn smusd() {
        unsafe {
            let a = i16x2::new(1, 2);
            let b = i16x2::new(5, 4);
            let r = dsp::smusd(::mem::transmute(a), ::mem::transmute(b));
            assert_eq!(r, -3);
        }
    }

    #[test]
    fn smusdx() {
        unsafe {
            let a = i16x2::new(1, 2);
            let b = i16x2::new(5, 4);
            let r = dsp::smusdx(::mem::transmute(a), ::mem::transmute(b));
            assert_eq!(r, -6);
        }
    }

    #[test]
    fn usad8() {
        unsafe {
            let a = i8x4::new(1, 2, 3, 4);
            let b = i8x4::new(4, 3, 2, 1);
            let r = dsp::usad8(::mem::transmute(a), ::mem::transmute(b));
            assert_eq!(r, 8);
        }
    }

    #[test]
    fn usad8a() {
        unsafe {
            let a = i8x4::new(1, 2, 3, 4);
            let b = i8x4::new(4, 3, 2, 1);
            let c = 10;
            let r = dsp::usad8a(::mem::transmute(a), ::mem::transmute(b), c);
            assert_eq!(r, 8 + c);
        }
    }
}
