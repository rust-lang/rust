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

    #[cfg_attr(not(target_feature = "mclass"), link_name = "llvm.arm.sel")]
    fn arm_sel(a: i32, b: i32) -> i32;
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
}
