//! Bit Manipulation Instruction (BMI) Set 1.0.
//!
//! The reference is [Intel 64 and IA-32 Architectures Software Developer's
//! Manual Volume 2: Instruction Set Reference, A-Z][intel64_ref].
//!
//! [Wikipedia][wikipedia_bmi] provides a quick overview of the instructions
//! available.
//!
//! [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf
//! [wikipedia_bmi]: https://en.wikipedia.org/wiki/Bit_Manipulation_Instruction_Sets#ABM_.28Advanced_Bit_Manipulation.29

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Extracts bits in range [`start`, `start` + `length`) from `a` into
/// the least significant bits of the result.
#[inline(always)]
#[target_feature = "+bmi"]
#[cfg_attr(test, assert_instr(bextr))]
pub unsafe fn _bextr_u32(a: u32, start: u32, len: u32) -> u32 {
    _bextr2_u32(a, (start & 0xff_u32) | ((len & 0xff_u32) << 8_u32))
}

/// Extracts bits in range [`start`, `start` + `length`) from `a` into
/// the least significant bits of the result.
#[inline(always)]
#[target_feature = "+bmi"]
#[cfg_attr(test, assert_instr(bextr))]
#[cfg(not(target_arch = "x86"))]
pub unsafe fn _bextr_u64(a: u64, start: u32, len: u32) -> u64 {
    _bextr2_u64(a, ((start & 0xff) | ((len & 0xff) << 8)) as u64)
}

/// Extracts bits of `a` specified by `control` into
/// the least significant bits of the result.
///
/// Bits [7,0] of `control` specify the index to the first bit in the range to
/// be extracted, and bits [15,8] specify the length of the range.
#[inline(always)]
#[target_feature = "+bmi"]
#[cfg_attr(test, assert_instr(bextr))]
pub unsafe fn _bextr2_u32(a: u32, control: u32) -> u32 {
    x86_bmi_bextr_32(a, control)
}

/// Extracts bits of `a` specified by `control` into
/// the least significant bits of the result.
///
/// Bits [7,0] of `control` specify the index to the first bit in the range to
/// be extracted, and bits [15,8] specify the length of the range.
#[inline(always)]
#[target_feature = "+bmi"]
#[cfg_attr(test, assert_instr(bextr))]
#[cfg(not(target_arch = "x86"))]
pub unsafe fn _bextr2_u64(a: u64, control: u64) -> u64 {
    x86_bmi_bextr_64(a, control)
}

/// Bitwise logical `AND` of inverted `a` with `b`.
#[inline(always)]
#[target_feature = "+bmi"]
#[cfg_attr(test, assert_instr(andn))]
pub unsafe fn _andn_u32(a: u32, b: u32) -> u32 {
    !a & b
}

/// Bitwise logical `AND` of inverted `a` with `b`.
#[inline(always)]
#[target_feature = "+bmi"]
#[cfg_attr(test, assert_instr(andn))]
pub unsafe fn _andn_u64(a: u64, b: u64) -> u64 {
    !a & b
}

/// Extract lowest set isolated bit.
#[inline(always)]
#[target_feature = "+bmi"]
#[cfg_attr(test, assert_instr(blsi))]
pub unsafe fn _blsi_u32(x: u32) -> u32 {
    x & x.wrapping_neg()
}

/// Extract lowest set isolated bit.
#[inline(always)]
#[target_feature = "+bmi"]
#[cfg_attr(test, assert_instr(blsi))]
#[cfg(not(target_arch = "x86"))] // generates lots of instructions
pub unsafe fn _blsi_u64(x: u64) -> u64 {
    x & x.wrapping_neg()
}

/// Get mask up to lowest set bit.
#[inline(always)]
#[target_feature = "+bmi"]
#[cfg_attr(test, assert_instr(blsmsk))]
pub unsafe fn _blsmsk_u32(x: u32) -> u32 {
    x ^ (x.wrapping_sub(1_u32))
}

/// Get mask up to lowest set bit.
#[inline(always)]
#[target_feature = "+bmi"]
#[cfg_attr(test, assert_instr(blsmsk))]
#[cfg(not(target_arch = "x86"))] // generates lots of instructions
pub unsafe fn _blsmsk_u64(x: u64) -> u64 {
    x ^ (x.wrapping_sub(1_u64))
}

/// Resets the lowest set bit of `x`.
///
/// If `x` is sets CF.
#[inline(always)]
#[target_feature = "+bmi"]
#[cfg_attr(test, assert_instr(blsr))]
pub unsafe fn _blsr_u32(x: u32) -> u32 {
    x & (x.wrapping_sub(1))
}

/// Resets the lowest set bit of `x`.
///
/// If `x` is sets CF.
#[inline(always)]
#[target_feature = "+bmi"]
#[cfg_attr(test, assert_instr(blsr))]
#[cfg(not(target_arch = "x86"))] // generates lots of instructions
pub unsafe fn _blsr_u64(x: u64) -> u64 {
    x & (x.wrapping_sub(1))
}

/// Counts the number of trailing least significant zero bits.
///
/// When the source operand is 0, it returns its size in bits.
#[inline(always)]
#[target_feature = "+bmi"]
#[cfg_attr(test, assert_instr(tzcnt))]
pub unsafe fn _tzcnt_u32(x: u32) -> u32 {
    x.trailing_zeros()
}

/// Counts the number of trailing least significant zero bits.
///
/// When the source operand is 0, it returns its size in bits.
#[inline(always)]
#[target_feature = "+bmi"]
#[cfg_attr(test, assert_instr(tzcnt))]
pub unsafe fn _tzcnt_u64(x: u64) -> u64 {
    x.trailing_zeros() as u64
}

/// Counts the number of trailing least significant zero bits.
///
/// When the source operand is 0, it returns its size in bits.
#[inline(always)]
#[target_feature = "+bmi"]
#[cfg_attr(test, assert_instr(tzcnt))]
pub unsafe fn _mm_tzcnt_32(x: u32) -> i32 {
    x.trailing_zeros() as i32
}

/// Counts the number of trailing least significant zero bits.
///
/// When the source operand is 0, it returns its size in bits.
#[inline(always)]
#[target_feature = "+bmi"]
#[cfg_attr(test, assert_instr(tzcnt))]
pub unsafe fn _mm_tzcnt_64(x: u64) -> i64 {
    x.trailing_zeros() as i64
}

#[allow(dead_code)]
extern "C" {
    #[link_name = "llvm.x86.bmi.bextr.32"]
    fn x86_bmi_bextr_32(x: u32, y: u32) -> u32;
    #[link_name = "llvm.x86.bmi.bextr.64"]
    fn x86_bmi_bextr_64(x: u64, y: u64) -> u64;
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use x86::i586::bmi;

    #[simd_test = "bmi"]
    unsafe fn _bextr_u32() {
        let r = bmi::_bextr_u32(0b0101_0000u32, 4, 4);
        assert_eq!(r, 0b0000_0101u32);
    }

    #[simd_test = "bmi"]
    #[cfg(not(target_arch = "x86"))]
    unsafe fn _bextr_u64() {
        let r = bmi::_bextr_u64(0b0101_0000u64, 4, 4);
        assert_eq!(r, 0b0000_0101u64);
    }

    #[simd_test = "bmi"]
    unsafe fn _andn_u32() {
        assert_eq!(bmi::_andn_u32(0, 0), 0);
        assert_eq!(bmi::_andn_u32(0, 1), 1);
        assert_eq!(bmi::_andn_u32(1, 0), 0);
        assert_eq!(bmi::_andn_u32(1, 1), 0);

        let r = bmi::_andn_u32(0b0000_0000u32, 0b0000_0000u32);
        assert_eq!(r, 0b0000_0000u32);

        let r = bmi::_andn_u32(0b0000_0000u32, 0b1111_1111u32);
        assert_eq!(r, 0b1111_1111u32);

        let r = bmi::_andn_u32(0b1111_1111u32, 0b0000_0000u32);
        assert_eq!(r, 0b0000_0000u32);

        let r = bmi::_andn_u32(0b1111_1111u32, 0b1111_1111u32);
        assert_eq!(r, 0b0000_0000u32);

        let r = bmi::_andn_u32(0b0100_0000u32, 0b0101_1101u32);
        assert_eq!(r, 0b0001_1101u32);
    }

    #[simd_test = "bmi"]
    #[cfg(not(target_arch = "x86"))]
    unsafe fn _andn_u64() {
        assert_eq!(bmi::_andn_u64(0, 0), 0);
        assert_eq!(bmi::_andn_u64(0, 1), 1);
        assert_eq!(bmi::_andn_u64(1, 0), 0);
        assert_eq!(bmi::_andn_u64(1, 1), 0);

        let r = bmi::_andn_u64(0b0000_0000u64, 0b0000_0000u64);
        assert_eq!(r, 0b0000_0000u64);

        let r = bmi::_andn_u64(0b0000_0000u64, 0b1111_1111u64);
        assert_eq!(r, 0b1111_1111u64);

        let r = bmi::_andn_u64(0b1111_1111u64, 0b0000_0000u64);
        assert_eq!(r, 0b0000_0000u64);

        let r = bmi::_andn_u64(0b1111_1111u64, 0b1111_1111u64);
        assert_eq!(r, 0b0000_0000u64);

        let r = bmi::_andn_u64(0b0100_0000u64, 0b0101_1101u64);
        assert_eq!(r, 0b0001_1101u64);
    }

    #[simd_test = "bmi"]
    unsafe fn _blsi_u32() {
        assert_eq!(bmi::_blsi_u32(0b1101_0000u32), 0b0001_0000u32);
    }

    #[simd_test = "bmi"]
    #[cfg(not(target_arch = "x86"))]
    unsafe fn _blsi_u64() {
        assert_eq!(bmi::_blsi_u64(0b1101_0000u64), 0b0001_0000u64);
    }

    #[simd_test = "bmi"]
    unsafe fn _blsmsk_u32() {
        let r = bmi::_blsmsk_u32(0b0011_0000u32);
        assert_eq!(r, 0b0001_1111u32);
    }

    #[simd_test = "bmi"]
    #[cfg(not(target_arch = "x86"))]
    unsafe fn _blsmsk_u64() {
        let r = bmi::_blsmsk_u64(0b0011_0000u64);
        assert_eq!(r, 0b0001_1111u64);
    }

    #[simd_test = "bmi"]
    unsafe fn _blsr_u32() {
        // TODO: test the behavior when the input is 0
        let r = bmi::_blsr_u32(0b0011_0000u32);
        assert_eq!(r, 0b0010_0000u32);
    }

    #[simd_test = "bmi"]
    #[cfg(not(target_arch = "x86"))]
    unsafe fn _blsr_u64() {
        // TODO: test the behavior when the input is 0
        let r = bmi::_blsr_u64(0b0011_0000u64);
        assert_eq!(r, 0b0010_0000u64);
    }

    #[simd_test = "bmi"]
    unsafe fn _tzcnt_u32() {
        assert_eq!(bmi::_tzcnt_u32(0b0000_0001u32), 0u32);
        assert_eq!(bmi::_tzcnt_u32(0b0000_0000u32), 32u32);
        assert_eq!(bmi::_tzcnt_u32(0b1001_0000u32), 4u32);
    }

    #[simd_test = "bmi"]
    #[cfg(not(target_arch = "x86"))]
    unsafe fn _tzcnt_u64() {
        assert_eq!(bmi::_tzcnt_u64(0b0000_0001u64), 0u64);
        assert_eq!(bmi::_tzcnt_u64(0b0000_0000u64), 64u64);
        assert_eq!(bmi::_tzcnt_u64(0b1001_0000u64), 4u64);
    }
}
