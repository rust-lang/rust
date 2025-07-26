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
use stdarch_test::assert_instr;

unsafe extern "C" {
    #[link_name = "llvm.x86.tbm.bextri.u64"]
    fn bextri_u64(a: u64, control: u64) -> u64;
}

/// Extracts bits of `a` specified by `control` into
/// the least significant bits of the result.
///
/// Bits `[7,0]` of `control` specify the index to the first bit in the range to
/// be extracted, and bits `[15,8]` specify the length of the range. For any bit
/// position in the specified range that lie beyond the MSB of the source operand,
/// zeroes will be written. If the range is empty, the result is zero.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(bextr, CONTROL = 0x0404))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86_updates", since = "1.82.0")]
pub unsafe fn _bextri_u64<const CONTROL: u64>(a: u64) -> u64 {
    static_assert_uimm_bits!(CONTROL, 16);
    unsafe { bextri_u64(a, CONTROL) }
}

/// Clears all bits below the least significant zero bit of `x`.
///
/// If there is no zero bit in `x`, it returns zero.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blcfill))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blcfill_u64(x: u64) -> u64 {
    x & x.wrapping_add(1)
}

/// Sets all bits of `x` to 1 except for the least significant zero bit.
///
/// If there is no zero bit in `x`, it sets all bits.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blci))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blci_u64(x: u64) -> u64 {
    x | !x.wrapping_add(1)
}

/// Sets the least significant zero bit of `x` and clears all other bits.
///
/// If there is no zero bit in `x`, it returns zero.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blcic))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blcic_u64(x: u64) -> u64 {
    !x & x.wrapping_add(1)
}

/// Sets the least significant zero bit of `x` and clears all bits above
/// that bit.
///
/// If there is no zero bit in `x`, it sets all the bits.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blcmsk))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blcmsk_u64(x: u64) -> u64 {
    x ^ x.wrapping_add(1)
}

/// Sets the least significant zero bit of `x`.
///
/// If there is no zero bit in `x`, it returns `x`.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blcs))]
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
pub unsafe fn _blsfill_u64(x: u64) -> u64 {
    x | x.wrapping_sub(1)
}

/// Clears least significant bit and sets all other bits.
///
/// If there is no set bit in `x`, it sets all the bits.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(blsic))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _blsic_u64(x: u64) -> u64 {
    !x | x.wrapping_sub(1)
}

/// Clears all bits below the least significant zero of `x` and sets all other
/// bits.
///
/// If the least significant bit of `x` is `0`, it sets all bits.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(t1mskc))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _t1mskc_u64(x: u64) -> u64 {
    !x | x.wrapping_add(1)
}

/// Sets all bits below the least significant one of `x` and clears all other
/// bits.
///
/// If the least significant bit of `x` is 1, it returns zero.
#[inline]
#[target_feature(enable = "tbm")]
#[cfg_attr(test, assert_instr(tzmsk))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _tzmsk_u64(x: u64) -> u64 {
    !x & x.wrapping_sub(1)
}

#[cfg(test)]
mod tests {
    use stdarch_test::simd_test;

    use crate::core_arch::x86_64::*;

    #[simd_test(enable = "tbm")]
    unsafe fn test_bextri_u64() {
        assert_eq!(_bextri_u64::<0x0404>(0b0101_0000u64), 0b0000_0101u64);
    }

    #[simd_test(enable = "tbm")]
    unsafe fn test_blcfill_u64() {
        assert_eq!(_blcfill_u64(0b0101_0111u64), 0b0101_0000u64);
        assert_eq!(_blcfill_u64(0b1111_1111u64), 0u64);
    }

    #[simd_test(enable = "tbm")]
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
    unsafe fn test_blcic_u64() {
        assert_eq!(_blcic_u64(0b0101_0001u64), 0b0000_0010u64);
        assert_eq!(_blcic_u64(0b1111_1111u64), 0b1_0000_0000u64);
    }

    #[simd_test(enable = "tbm")]
    unsafe fn test_blcmsk_u64() {
        assert_eq!(_blcmsk_u64(0b0101_0001u64), 0b0000_0011u64);
        assert_eq!(_blcmsk_u64(0b1111_1111u64), 0b1_1111_1111u64);
    }

    #[simd_test(enable = "tbm")]
    unsafe fn test_blcs_u64() {
        assert_eq!(_blcs_u64(0b0101_0001u64), 0b0101_0011u64);
        assert_eq!(_blcs_u64(0b1111_1111u64), 0b1_1111_1111u64);
    }

    #[simd_test(enable = "tbm")]
    unsafe fn test_blsfill_u64() {
        assert_eq!(_blsfill_u64(0b0101_0100u64), 0b0101_0111u64);
        assert_eq!(
            _blsfill_u64(0u64),
            0b1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64
        );
    }

    #[simd_test(enable = "tbm")]
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
    unsafe fn test_t1mskc_u64() {
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
    unsafe fn test_tzmsk_u64() {
        assert_eq!(_tzmsk_u64(0b0101_1000u64), 0b0000_0111u64);
        assert_eq!(_tzmsk_u64(0b0101_1001u64), 0b0000_0000u64);
    }
}
