//! Bit Manipulation Instruction (BMI) Set 1.0.
//!
//! The reference is [Intel 64 and IA-32 Architectures Software Developer's
//! Manual Volume 2: Instruction Set Reference,
//! A-Z](http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf).
//!
//! [Wikipedia](https://en.wikipedia.org/wiki/Bit_Manipulation_Instruction_Sets#BMI1_.28Bit_Manipulation_Instruction_Set_1.29)
//! provides a quick overview of the available instructions.

#[allow(dead_code)]
extern "C" {
    #[link_name="llvm.x86.bmi.bextr.32"]
    fn x86_bmi_bextr_32(x: u32, y: u32) -> u32;
    #[link_name="llvm.x86.bmi.bextr.64"]
    fn x86_bmi_bextr_64(x: u64, y: u64) -> u64;
}

/// Extracts bits in range [`start`, `start` + `length`) from `a` into
/// the least significant bits of the result.
#[inline(always)]
#[target_feature = "+bmi"]
pub fn _bextr_u32(a: u32, start: u32, len: u32) -> u32 {
    _bextr2_u32(a, (start & 0xffu32) | ((len & 0xffu32) << 8u32))
}

/// Extracts bits in range [`start`, `start` + `length`) from `a` into
/// the least significant bits of the result.
#[inline(always)]
#[target_feature = "+bmi"]
pub fn _bextr_u64(a: u64, start: u64, len: u64) -> u64 {
    _bextr2_u64(a, (start & 0xffu64) | ((len & 0xffu64) << 8u64))
}

/// Extracts bits of `a` specified by `control` into
/// the least significant bits of the result.
///
/// Bits [7,0] of `control` specify the index to the first bit in the range to be
/// extracted, and bits [15,8] specify the length of the range.
#[inline(always)]
#[target_feature = "+bmi"]
pub fn _bextr2_u32(a: u32, control: u32) -> u32 {
    unsafe { x86_bmi_bextr_32(a, control) }
}

/// Extracts bits of `a` specified by `control` into
/// the least significant bits of the result.
///
/// Bits [7,0] of `control` specify the index to the first bit in the range to be
/// extracted, and bits [15,8] specify the length of the range.
#[inline(always)]
#[target_feature = "+bmi"]
pub fn _bextr2_u64(a: u64, control: u64) -> u64 {
    unsafe { x86_bmi_bextr_64(a, control) }
}

/// Bitwise logical `AND` of inverted `a` with `b`.
#[inline(always)]
#[target_feature = "+bmi"]
pub fn _andn_u32(a: u32, b: u32) -> u32 {
    !a & b
}

/// Bitwise logical `AND` of inverted `a` with `b`.
#[inline(always)]
#[target_feature = "+bmi"]
pub fn _andn_u64(a: u64, b: u64) -> u64 {
    !a & b
}

/// Extract lowest set isolated bit.
#[inline(always)]
#[target_feature = "+bmi"]
pub fn _blsi_u32(x: u32) -> u32 {
    x & x.wrapping_neg()
}

/// Extract lowest set isolated bit.
#[inline(always)]
#[target_feature = "+bmi"]
pub fn _blsi_u64(x: u64) -> u64 {
    x & x.wrapping_neg()
}

/// Get mask up to lowest set bit.
#[inline(always)]
#[target_feature = "+bmi"]
pub fn _blsmsk_u32(x: u32) -> u32 {
    x ^ (x.wrapping_sub(1u32))
}

/// Get mask up to lowest set bit.
#[inline(always)]
#[target_feature = "+bmi"]
pub fn _blsmsk_u64(x: u64) -> u64 {
    x ^ (x.wrapping_sub(1u64))
}

/// Resets the lowest set bit of `x`.
///
/// If `x` is sets CF.
#[inline(always)]
#[target_feature = "+bmi"]
pub fn _blsr_u32(x: u32) -> u32 {
    x & (x.wrapping_sub(1))
}

/// Resets the lowest set bit of `x`.
///
/// If `x` is sets CF.
#[inline(always)]
#[target_feature = "+bmi"]
pub fn _blsr_u64(x: u64) -> u64 {
    x & (x.wrapping_sub(1))
}

/// Counts the number of trailing least significant zero bits.
///
/// When the source operand is 0, it returns its size in bits.
#[inline(always)]
#[target_feature = "+bmi"]
pub fn _tzcnt_u16(x: u16) -> u16 {
    x.trailing_zeros() as u16
}

/// Counts the number of trailing least significant zero bits.
///
/// When the source operand is 0, it returns its size in bits.
#[inline(always)]
#[target_feature = "+bmi"]
pub fn _tzcnt_u32(x: u32) -> u32 {
    x.trailing_zeros()
}

/// Counts the number of trailing least significant zero bits.
///
/// When the source operand is 0, it returns its size in bits.
#[inline(always)]
#[target_feature = "+bmi"]
pub fn _tzcnt_u64(x: u64) -> u64 {
    x.trailing_zeros() as u64
}

/// Counts the number of trailing least significant zero bits.
///
/// When the source operand is 0, it returns its size in bits.
#[inline(always)]
#[target_feature = "+bmi"]
pub fn _mm_tzcnt_u32(x: u32) -> u32 {
    x.trailing_zeros()
}

/// Counts the number of trailing least significant zero bits.
///
/// When the source operand is 0, it returns its size in bits.
#[inline(always)]
#[target_feature = "+bmi"]
pub fn _mm_tzcnt_u64(x: u64) -> u64 {
    x.trailing_zeros() as u64
}

#[cfg(all(test, target_feature = "bmi", any(target_arch = "x86", target_arch = "x86_64")))]
mod tests {
    use x86::bmi;

    #[test]
    #[target_feature = "+bmi"]
    fn _bextr_u32() {
        assert_eq!(bmi::_bextr_u32(0b0101_0000u32, 4, 4), 0b0000_0101u32);
    }

    #[test]
    #[target_feature = "+bmi"]
    fn _bextr_u64() {
        assert_eq!(bmi::_bextr_u64(0b0101_0000u64, 4, 4), 0b0000_0101u64);
    }

    #[test]
    #[target_feature = "+bmi"]
    fn _andn_u32() {
        assert_eq!(bmi::_andn_u32(0, 0), 0);
        assert_eq!(bmi::_andn_u32(0, 1), 1);
        assert_eq!(bmi::_andn_u32(1, 0), 0);
        assert_eq!(bmi::_andn_u32(1, 1), 0);

        assert_eq!(bmi::_andn_u32(0b0000_0000u32, 0b0000_0000u32), 0b0000_0000u32);
        assert_eq!(bmi::_andn_u32(0b0000_0000u32, 0b1111_1111u32), 0b1111_1111u32);
        assert_eq!(bmi::_andn_u32(0b1111_1111u32, 0b0000_0000u32), 0b0000_0000u32);
        assert_eq!(bmi::_andn_u32(0b1111_1111u32, 0b1111_1111u32), 0b0000_0000u32);
        assert_eq!(bmi::_andn_u32(0b0100_0000u32, 0b0101_1101u32), 0b0001_1101u32);
    }

    #[test]
    #[target_feature = "+bmi"]
    fn _andn_u64() {
        assert_eq!(bmi::_andn_u64(0, 0), 0);
        assert_eq!(bmi::_andn_u64(0, 1), 1);
        assert_eq!(bmi::_andn_u64(1, 0), 0);
        assert_eq!(bmi::_andn_u64(1, 1), 0);

        assert_eq!(bmi::_andn_u64(0b0000_0000u64, 0b0000_0000u64), 0b0000_0000u64);
        assert_eq!(bmi::_andn_u64(0b0000_0000u64, 0b1111_1111u64), 0b1111_1111u64);
        assert_eq!(bmi::_andn_u64(0b1111_1111u64, 0b0000_0000u64), 0b0000_0000u64);
        assert_eq!(bmi::_andn_u64(0b1111_1111u64, 0b1111_1111u64), 0b0000_0000u64);
        assert_eq!(bmi::_andn_u64(0b0100_0000u64, 0b0101_1101u64), 0b0001_1101u64);
    }

    #[test]
    #[target_feature = "+bmi"]
    fn _blsi_u32() {
        assert_eq!(bmi::_blsi_u32(0b1101_0000u32), 0b0001_0000u32);
    }

    #[test]
    #[target_feature = "+bmi"]
    fn _blsi_u64() {
        assert_eq!(bmi::_blsi_u64(0b1101_0000u64), 0b0001_0000u64);
    }

    #[test]
    #[target_feature = "+bmi"]
    fn _blsmsk_u32() {
        assert_eq!(bmi::_blsmsk_u32(0b0011_0000u32), 0b0001_1111u32);
    }

    #[test]
    #[target_feature = "+bmi"]
    fn _blsmsk_u64() {
        assert_eq!(bmi::_blsmsk_u64(0b0011_0000u64), 0b0001_1111u64);
    }

    #[test]
    #[target_feature = "+bmi"]
    fn _blsr_u32() {
        /// TODO: test the behavior when the input is 0
        assert_eq!(bmi::_blsr_u32(0b0011_0000u32), 0b0010_0000u32);
    }

    #[test]
    #[target_feature = "+bmi"]
    fn _blsr_u64() {
        /// TODO: test the behavior when the input is 0
        assert_eq!(bmi::_blsr_u64(0b0011_0000u64), 0b0010_0000u64);
    }

    #[test]
    #[target_feature = "+bmi"]
    fn _tzcnt_u16() {
        assert_eq!(bmi::_tzcnt_u16(0b0000_0001u16), 0u16);
        assert_eq!(bmi::_tzcnt_u16(0b0000_0000u16), 16u16);
        assert_eq!(bmi::_tzcnt_u16(0b1001_0000u16), 4u16);
    }

    #[test]
    #[target_feature = "+bmi"]
    fn _tzcnt_u32() {
        assert_eq!(bmi::_tzcnt_u32(0b0000_0001u32), 0u32);
        assert_eq!(bmi::_tzcnt_u32(0b0000_0000u32), 32u32);
        assert_eq!(bmi::_tzcnt_u32(0b1001_0000u32), 4u32);
    }

    #[test]
    #[target_feature = "+bmi"]
    fn _tzcnt_u64() {
        assert_eq!(bmi::_tzcnt_u64(0b0000_0001u64), 0u64);
        assert_eq!(bmi::_tzcnt_u64(0b0000_0000u64), 64u64);
        assert_eq!(bmi::_tzcnt_u64(0b1001_0000u64), 4u64);
    }
}
