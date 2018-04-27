//! Trailing Bit Manipulation (TBM) instruction set.
//!
//! The reference is [AMD64 Architecture Programmer's Manual, Volume 3:
//! General-Purpose and System Instructions][amd64_ref].
//!
//! [Wikipedia][wikipedia_bmi] provides a quick overview of the available
//! instructions.
//!
//! [amd64_ref]: http://support.amd.com/TechDocs/24594.pdf
//! [wikipedia_bmi]:
//! https://en.wikipedia.org/wiki/Bit_Manipulation_Instruction_Sets#ABM_.28Advanced_Bit_Manipulation.29

#[cfg(test)]
use stdsimd_test::assert_instr;

// FIXME(blocked on #248)
// TODO: LLVM-CODEGEN ERROR: LLVM ERROR: Cannot select:
// intrinsic %llvm.x86.tbm.bextri.u32
/*
#[allow(dead_code)]
extern "C" {
    #[link_name="llvm.x86.tbm.bextri.u32"]
    fn x86_tbm_bextri_u32(a: u32, y: u32) -> u32;
    #[link_name="llvm.x86.tbm.bextri.u64"]
    fn x86_tbm_bextri_u64(x: u64, y: u64) -> u64;
}

/// Extracts bits in range [`start`, `start` + `length`) from `a` into
/// the least significant bits of the result.
#[inline]
#[target_feature(enable = "tbm")]
pub fn _bextr_u32(a: u32, start: u32, len: u32) -> u32 {
    _bextr2_u32(a, (start & 0xffu32) | ((len & 0xffu32) << 8u32))
}

/// Extracts bits in range [`start`, `start` + `length`) from `a` into
/// the least significant bits of the result.
#[inline]
#[target_feature(enable = "tbm")]
pub fn _bextr_u64(a: u64, start: u64, len: u64) -> u64 {
    _bextr2_u64(a, (start & 0xffu64) | ((len & 0xffu64) << 8u64))
}

/// Extracts bits of `a` specified by `control` into
/// the least significant bits of the result.
///
/// Bits `[7,0]` of `control` specify the index to the first bit in the range to
/// be extracted, and bits `[15,8]` specify the length of the range.
#[inline]
#[target_feature(enable = "tbm")]
pub fn _bextr2_u32(a: u32, control: u32) -> u32 {
    unsafe { x86_tbm_bextri_u32(a, control) }
}

/// Extracts bits of `a` specified by `control` into
/// the least significant bits of the result.
///
/// Bits `[7,0]` of `control` specify the index to the first bit in the range to
/// be extracted, and bits `[15,8]` specify the length of the range.
#[inline]
#[target_feature(enable = "tbm")]
pub fn _bextr2_u64(a: u64, control: u64) -> u64 {
    unsafe { x86_tbm_bextri_u64(a, control) }
}
*/

/// Clears all bits below the least significant zero bit of `x`.
///
/// If there is no zero bit in `x`, it returns zero.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blcfill))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blcfill_u32(x: u32) -> u32 {
    x & (x.wrapping_add(1))
}

/// Clears all bits below the least significant zero bit of `x`.
///
/// If there is no zero bit in `x`, it returns zero.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blcfill))]
#[cfg(not(target_arch = "x86"))] // generates lots of instructions
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blcfill_u64(x: u64) -> u64 {
    x & (x.wrapping_add(1))
}

/// Sets all bits of `x` to 1 except for the least significant zero bit.
///
/// If there is no zero bit in `x`, it sets all bits.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blci))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blci_u32(x: u32) -> u32 {
    x | !(x.wrapping_add(1))
}

/// Sets all bits of `x` to 1 except for the least significant zero bit.
///
/// If there is no zero bit in `x`, it sets all bits.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blci))]
#[cfg(not(target_arch = "x86"))] // generates lots of instructions
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blci_u64(x: u64) -> u64 {
    x | !(x.wrapping_add(1))
}

/// Sets the least significant zero bit of `x` and clears all other bits.
///
/// If there is no zero bit in `x`, it returns zero.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blcic))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blcic_u32(x: u32) -> u32 {
    !x & (x.wrapping_add(1))
}

/// Sets the least significant zero bit of `x` and clears all other bits.
///
/// If there is no zero bit in `x`, it returns zero.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blcic))]
#[cfg(not(target_arch = "x86"))] // generates lots of instructions
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blcic_u64(x: u64) -> u64 {
    !x & (x.wrapping_add(1))
}

/// Sets the least significant zero bit of `x` and clears all bits above
/// that bit.
///
/// If there is no zero bit in `x`, it sets all the bits.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blcmsk))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blcmsk_u32(x: u32) -> u32 {
    x ^ (x.wrapping_add(1))
}

/// Sets the least significant zero bit of `x` and clears all bits above
/// that bit.
///
/// If there is no zero bit in `x`, it sets all the bits.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blcmsk))]
#[cfg(not(target_arch = "x86"))] // generates lots of instructions
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blcmsk_u64(x: u64) -> u64 {
    x ^ (x.wrapping_add(1))
}

/// Sets the least significant zero bit of `x`.
///
/// If there is no zero bit in `x`, it returns `x`.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blcs))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blcs_u32(x: u32) -> u32 {
    x | (x.wrapping_add(1))
}

/// Sets the least significant zero bit of `x`.
///
/// If there is no zero bit in `x`, it returns `x`.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blcs))]
#[cfg(not(target_arch = "x86"))] // generates lots of instructions
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blcs_u64(x: u64) -> u64 {
    x | x.wrapping_add(1)
}

/// Sets all bits of `x` below the least significant one.
///
/// If there is no set bit in `x`, it sets all the bits.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blsfill))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blsfill_u32(x: u32) -> u32 {
    x | (x.wrapping_sub(1))
}

/// Sets all bits of `x` below the least significant one.
///
/// If there is no set bit in `x`, it sets all the bits.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blsfill))]
#[cfg(not(target_arch = "x86"))] // generates lots of instructions
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blsfill_u64(x: u64) -> u64 {
    x | (x.wrapping_sub(1))
}

/// Clears least significant bit and sets all other bits.
///
/// If there is no set bit in `x`, it sets all the bits.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blsic))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blsic_u32(x: u32) -> u32 {
    !x | (x.wrapping_sub(1))
}

/// Clears least significant bit and sets all other bits.
///
/// If there is no set bit in `x`, it sets all the bits.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blsic))]
#[cfg(not(target_arch = "x86"))] // generates lots of instructions
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blsic_u64(x: u64) -> u64 {
    !x | (x.wrapping_sub(1))
}

/// Clears all bits below the least significant zero of `x` and sets all other
/// bits.
///
/// If the least significant bit of `x` is 0, it sets all bits.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(t1mskc))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _t1mskc_u32(x: u32) -> u32 {
    !x | (x.wrapping_add(1))
}

/// Clears all bits below the least significant zero of `x` and sets all other
/// bits.
///
/// If the least significant bit of `x` is 0, it sets all bits.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(t1mskc))]
#[cfg(not(target_arch = "x86"))] // generates lots of instructions
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _t1mskc_u64(x: u64) -> u64 {
    !x | (x.wrapping_add(1))
}

/// Sets all bits below the least significant one of `x` and clears all other
/// bits.
///
/// If the least significant bit of `x` is 1, it returns zero.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(tzmsk))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _tzmsk_u32(x: u32) -> u32 {
    !x & (x.wrapping_sub(1))
}

/// Sets all bits below the least significant one of `x` and clears all other
/// bits.
///
/// If the least significant bit of `x` is 1, it returns zero.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(tzmsk))]
#[cfg(not(target_arch = "x86"))] // generates lots of instructions
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _tzmsk_u64(x: u64) -> u64 {
    !x & (x.wrapping_sub(1))
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use coresimd::x86::*;

    /*
    #[simd_test(enable = "tbm")]
    unsafe fn test_bextr_u32() {
        assert_eq!(_bextr_u32(0b0101_0000u32, 4, 4), 0b0000_0101u32);
    }

    #[simd_test(enable = "tbm")]
    unsafe fn test_bextr_u64() {
        assert_eq!(_bextr_u64(0b0101_0000u64, 4, 4), 0b0000_0101u64);
    }
    */

    #[simd_test(enable = "tbm")]
    unsafe fn test_blcfill_u32() {
        assert_eq!(_blcfill_u32(0b0101_0111u32), 0b0101_0000u32);
        assert_eq!(_blcfill_u32(0b1111_1111u32), 0u32);
    }

    #[simd_test(enable = "tbm")]
    #[cfg(not(target_arch = "x86"))]
    unsafe fn test_blcfill_u64() {
        assert_eq!(_blcfill_u64(0b0101_0111u64), 0b0101_0000u64);
        assert_eq!(_blcfill_u64(0b1111_1111u64), 0u64);
    }

    #[simd_test(enable = "tbm")]
    unsafe fn test_blci_u32() {
        assert_eq!(
            _blci_u32(0b0101_0000u32),
            0b1111_1111_1111_1111_1111_1111_1111_1110u32
        );
        assert_eq!(
            _blci_u32(0b1111_1111u32),
            0b1111_1111_1111_1111_1111_1110_1111_1111u32
        );
    }

    #[simd_test(enable = "tbm")]
    #[cfg(not(target_arch = "x86"))]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    unsafe fn test_blci_u64() {
        assert_eq!(
            _blci_u64(0b0101_0000u64),
            0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1110u64
        );
        assert_eq!(
            _blci_u64(0b1111_1111u64),
            0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1110_1111_1111u64
        );
    }

    #[simd_test(enable = "tbm")]
    unsafe fn test_blcic_u32() {
        assert_eq!(_blcic_u32(0b0101_0001u32), 0b0000_0010u32);
        assert_eq!(_blcic_u32(0b1111_1111u32), 0b1_0000_0000u32);
    }

    #[simd_test(enable = "tbm")]
    #[cfg(not(target_arch = "x86"))]
    unsafe fn test_blcic_u64() {
        assert_eq!(_blcic_u64(0b0101_0001u64), 0b0000_0010u64);
        assert_eq!(_blcic_u64(0b1111_1111u64), 0b1_0000_0000u64);
    }

    #[simd_test(enable = "tbm")]
    unsafe fn test_blcmsk_u32() {
        assert_eq!(_blcmsk_u32(0b0101_0001u32), 0b0000_0011u32);
        assert_eq!(_blcmsk_u32(0b1111_1111u32), 0b1_1111_1111u32);
    }

    #[simd_test(enable = "tbm")]
    #[cfg(not(target_arch = "x86"))]
    unsafe fn test_blcmsk_u64() {
        assert_eq!(_blcmsk_u64(0b0101_0001u64), 0b0000_0011u64);
        assert_eq!(_blcmsk_u64(0b1111_1111u64), 0b1_1111_1111u64);
    }

    #[simd_test(enable = "tbm")]
    unsafe fn test_blcs_u32() {
        assert_eq!(_blcs_u32(0b0101_0001u32), 0b0101_0011u32);
        assert_eq!(_blcs_u32(0b1111_1111u32), 0b1_1111_1111u32);
    }

    #[simd_test(enable = "tbm")]
    #[cfg(not(target_arch = "x86"))]
    unsafe fn test_blcs_u64() {
        assert_eq!(_blcs_u64(0b0101_0001u64), 0b0101_0011u64);
        assert_eq!(_blcs_u64(0b1111_1111u64), 0b1_1111_1111u64);
    }

    #[simd_test(enable = "tbm")]
    unsafe fn test_blsfill_u32() {
        assert_eq!(_blsfill_u32(0b0101_0100u32), 0b0101_0111u32);
        assert_eq!(
            _blsfill_u32(0u32),
            0b1111_1111_1111_1111_1111_1111_1111_1111u32
        );
    }

    #[simd_test(enable = "tbm")]
    #[cfg(not(target_arch = "x86"))]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    unsafe fn test_blsfill_u64() {
        assert_eq!(_blsfill_u64(0b0101_0100u64), 0b0101_0111u64);
        assert_eq!(
            _blsfill_u64(0u64),
            0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64
        );
    }

    #[simd_test(enable = "tbm")]
    unsafe fn test_blsic_u32() {
        assert_eq!(
            _blsic_u32(0b0101_0100u32),
            0b1111_1111_1111_1111_1111_1111_1111_1011u32
        );
        assert_eq!(
            _blsic_u32(0u32),
            0b1111_1111_1111_1111_1111_1111_1111_1111u32
        );
    }

    #[simd_test(enable = "tbm")]
    #[cfg(not(target_arch = "x86"))]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    unsafe fn test_blsic_u64() {
        assert_eq!(
            _blsic_u64(0b0101_0100u64),
            0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1011u64
        );
        assert_eq!(
            _blsic_u64(0u64),
            0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64
        );
    }

    #[simd_test(enable = "tbm")]
    unsafe fn test_t1mskc_u32() {
        assert_eq!(
            _t1mskc_u32(0b0101_0111u32),
            0b1111_1111_1111_1111_1111_1111_1111_1000u32
        );
        assert_eq!(
            _t1mskc_u32(0u32),
            0b1111_1111_1111_1111_1111_1111_1111_1111u32
        );
    }

    #[simd_test(enable = "tbm")]
    #[cfg(not(target_arch = "x86"))]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    unsafe fn test_t1mksc_u64() {
        assert_eq!(
            _t1mskc_u64(0b0101_0111u64),
            0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1000u64
        );
        assert_eq!(
            _t1mskc_u64(0u64),
            0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64
        );
    }

    #[simd_test(enable = "tbm")]
    unsafe fn test_tzmsk_u32() {
        assert_eq!(_tzmsk_u32(0b0101_1000u32), 0b0000_0111u32);
        assert_eq!(_tzmsk_u32(0b0101_1001u32), 0b0000_0000u32);
    }

    #[simd_test(enable = "tbm")]
    #[cfg(not(target_arch = "x86"))]
    unsafe fn test_tzmsk_u64() {
        assert_eq!(_tzmsk_u64(0b0101_1000u64), 0b0000_0111u64);
        assert_eq!(_tzmsk_u64(0b0101_1001u64), 0b0000_0000u64);
    }
}
