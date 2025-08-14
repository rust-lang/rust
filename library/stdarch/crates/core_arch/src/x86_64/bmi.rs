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
use stdarch_test::assert_instr;

/// Extracts bits in range [`start`, `start` + `length`) from `a` into
/// the least significant bits of the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_bextr_u64)
#[inline]
#[target_feature(enable = "bmi1")]
#[cfg_attr(test, assert_instr(bextr))]
#[cfg(not(target_arch = "x86"))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _bextr_u64(a: u64, start: u32, len: u32) -> u64 {
    _bextr2_u64(a, ((start & 0xff) | ((len & 0xff) << 8)) as u64)
}

/// Extracts bits of `a` specified by `control` into
/// the least significant bits of the result.
///
/// Bits `[7,0]` of `control` specify the index to the first bit in the range
/// to be extracted, and bits `[15,8]` specify the length of the range.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_bextr2_u64)
#[inline]
#[target_feature(enable = "bmi1")]
#[cfg_attr(test, assert_instr(bextr))]
#[cfg(not(target_arch = "x86"))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _bextr2_u64(a: u64, control: u64) -> u64 {
    unsafe { x86_bmi_bextr_64(a, control) }
}

/// Bitwise logical `AND` of inverted `a` with `b`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_andn_u64)
#[inline]
#[target_feature(enable = "bmi1")]
#[cfg_attr(test, assert_instr(andn))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _andn_u64(a: u64, b: u64) -> u64 {
    !a & b
}

/// Extracts lowest set isolated bit.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_blsi_u64)
#[inline]
#[target_feature(enable = "bmi1")]
#[cfg_attr(test, assert_instr(blsi))]
#[cfg(not(target_arch = "x86"))] // generates lots of instructions
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _blsi_u64(x: u64) -> u64 {
    x & x.wrapping_neg()
}

/// Gets mask up to lowest set bit.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_blsmsk_u64)
#[inline]
#[target_feature(enable = "bmi1")]
#[cfg_attr(test, assert_instr(blsmsk))]
#[cfg(not(target_arch = "x86"))] // generates lots of instructions
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _blsmsk_u64(x: u64) -> u64 {
    x ^ (x.wrapping_sub(1_u64))
}

/// Resets the lowest set bit of `x`.
///
/// If `x` is sets CF.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_blsr_u64)
#[inline]
#[target_feature(enable = "bmi1")]
#[cfg_attr(test, assert_instr(blsr))]
#[cfg(not(target_arch = "x86"))] // generates lots of instructions
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _blsr_u64(x: u64) -> u64 {
    x & (x.wrapping_sub(1))
}

/// Counts the number of trailing least significant zero bits.
///
/// When the source operand is `0`, it returns its size in bits.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_tzcnt_u64)
#[inline]
#[target_feature(enable = "bmi1")]
#[cfg_attr(test, assert_instr(tzcnt))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _tzcnt_u64(x: u64) -> u64 {
    x.trailing_zeros() as u64
}

/// Counts the number of trailing least significant zero bits.
///
/// When the source operand is `0`, it returns its size in bits.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_tzcnt_64)
#[inline]
#[target_feature(enable = "bmi1")]
#[cfg_attr(test, assert_instr(tzcnt))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_tzcnt_64(x: u64) -> i64 {
    x.trailing_zeros() as i64
}

unsafe extern "C" {
    #[link_name = "llvm.x86.bmi.bextr.64"]
    fn x86_bmi_bextr_64(x: u64, y: u64) -> u64;
}

#[cfg(test)]
mod tests {
    use stdarch_test::simd_test;

    use crate::core_arch::{x86::*, x86_64::*};

    #[simd_test(enable = "bmi1")]
    unsafe fn test_bextr_u64() {
        let r = _bextr_u64(0b0101_0000u64, 4, 4);
        assert_eq!(r, 0b0000_0101u64);
    }

    #[simd_test(enable = "bmi1")]
    unsafe fn test_andn_u64() {
        assert_eq!(_andn_u64(0, 0), 0);
        assert_eq!(_andn_u64(0, 1), 1);
        assert_eq!(_andn_u64(1, 0), 0);
        assert_eq!(_andn_u64(1, 1), 0);

        let r = _andn_u64(0b0000_0000u64, 0b0000_0000u64);
        assert_eq!(r, 0b0000_0000u64);

        let r = _andn_u64(0b0000_0000u64, 0b1111_1111u64);
        assert_eq!(r, 0b1111_1111u64);

        let r = _andn_u64(0b1111_1111u64, 0b0000_0000u64);
        assert_eq!(r, 0b0000_0000u64);

        let r = _andn_u64(0b1111_1111u64, 0b1111_1111u64);
        assert_eq!(r, 0b0000_0000u64);

        let r = _andn_u64(0b0100_0000u64, 0b0101_1101u64);
        assert_eq!(r, 0b0001_1101u64);
    }

    #[simd_test(enable = "bmi1")]
    unsafe fn test_blsi_u64() {
        assert_eq!(_blsi_u64(0b1101_0000u64), 0b0001_0000u64);
    }

    #[simd_test(enable = "bmi1")]
    unsafe fn test_blsmsk_u64() {
        let r = _blsmsk_u64(0b0011_0000u64);
        assert_eq!(r, 0b0001_1111u64);
    }

    #[simd_test(enable = "bmi1")]
    unsafe fn test_blsr_u64() {
        // TODO: test the behavior when the input is `0`.
        let r = _blsr_u64(0b0011_0000u64);
        assert_eq!(r, 0b0010_0000u64);
    }

    #[simd_test(enable = "bmi1")]
    unsafe fn test_tzcnt_u64() {
        assert_eq!(_tzcnt_u64(0b0000_0001u64), 0u64);
        assert_eq!(_tzcnt_u64(0b0000_0000u64), 64u64);
        assert_eq!(_tzcnt_u64(0b1001_0000u64), 4u64);
    }
}
