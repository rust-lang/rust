//! Bit Manipulation Instruction (BMI) Set 2.0.
//!
//! The reference is [Intel 64 and IA-32 Architectures Software Developer's
//! Manual Volume 2: Instruction Set Reference, A-Z][intel64_ref].
//!
//! [Wikipedia][wikipedia_bmi] provides a quick overview of the instructions
//! available.
//!
//! [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf
//! [wikipedia_bmi]:
//! https://en.wikipedia.org/wiki/Bit_Manipulation_Instruction_Sets#ABM_.28Advanced_Bit_Manipulation.29

#[cfg(test)]
use stdarch_test::assert_instr;

/// Unsigned multiply without affecting flags.
///
/// Unsigned multiplication of `a` with `b` returning a pair `(lo, hi)` with
/// the low half and the high half of the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mulx_u64)
#[inline]
#[cfg_attr(test, assert_instr(mul))]
#[target_feature(enable = "bmi2")]
#[cfg(not(target_arch = "x86"))] // calls an intrinsic
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mulx_u64(a: u64, b: u64, hi: &mut u64) -> u64 {
    let result: u128 = (a as u128) * (b as u128);
    *hi = (result >> 64) as u64;
    result as u64
}

/// Zeroes higher bits of `a` >= `index`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_bzhi_u64)
#[inline]
#[target_feature(enable = "bmi2")]
#[cfg_attr(test, assert_instr(bzhi))]
#[cfg(not(target_arch = "x86"))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _bzhi_u64(a: u64, index: u32) -> u64 {
    unsafe { x86_bmi2_bzhi_64(a, index as u64) }
}

/// Scatter contiguous low order bits of `a` to the result at the positions
/// specified by the `mask`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_pdep_u64)
#[inline]
#[target_feature(enable = "bmi2")]
#[cfg_attr(test, assert_instr(pdep))]
#[cfg(not(target_arch = "x86"))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _pdep_u64(a: u64, mask: u64) -> u64 {
    unsafe { x86_bmi2_pdep_64(a, mask) }
}

/// Gathers the bits of `x` specified by the `mask` into the contiguous low
/// order bit positions of the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_pext_u64)
#[inline]
#[target_feature(enable = "bmi2")]
#[cfg_attr(test, assert_instr(pext))]
#[cfg(not(target_arch = "x86"))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _pext_u64(a: u64, mask: u64) -> u64 {
    unsafe { x86_bmi2_pext_64(a, mask) }
}

unsafe extern "C" {
    #[link_name = "llvm.x86.bmi.bzhi.64"]
    fn x86_bmi2_bzhi_64(x: u64, y: u64) -> u64;
    #[link_name = "llvm.x86.bmi.pdep.64"]
    fn x86_bmi2_pdep_64(x: u64, y: u64) -> u64;
    #[link_name = "llvm.x86.bmi.pext.64"]
    fn x86_bmi2_pext_64(x: u64, y: u64) -> u64;
}

#[cfg(test)]
mod tests {
    use stdarch_test::simd_test;

    use crate::core_arch::x86_64::*;

    #[simd_test(enable = "bmi2")]
    unsafe fn test_pext_u64() {
        let n = 0b1011_1110_1001_0011u64;

        let m0 = 0b0110_0011_1000_0101u64;
        let s0 = 0b0000_0000_0011_0101u64;

        let m1 = 0b1110_1011_1110_1111u64;
        let s1 = 0b0001_0111_0100_0011u64;

        assert_eq!(_pext_u64(n, m0), s0);
        assert_eq!(_pext_u64(n, m1), s1);
    }

    #[simd_test(enable = "bmi2")]
    unsafe fn test_pdep_u64() {
        let n = 0b1011_1110_1001_0011u64;

        let m0 = 0b0110_0011_1000_0101u64;
        let s0 = 0b0000_0010_0000_0101u64;

        let m1 = 0b1110_1011_1110_1111u64;
        let s1 = 0b1110_1001_0010_0011u64;

        assert_eq!(_pdep_u64(n, m0), s0);
        assert_eq!(_pdep_u64(n, m1), s1);
    }

    #[simd_test(enable = "bmi2")]
    unsafe fn test_bzhi_u64() {
        let n = 0b1111_0010u64;
        let s = 0b0001_0010u64;
        assert_eq!(_bzhi_u64(n, 5), s);
    }

    #[simd_test(enable = "bmi2")]
    #[rustfmt::skip]
    unsafe fn test_mulx_u64() {
        let a: u64 = 9_223_372_036_854_775_800;
        let b: u64 = 100;
        let mut hi = 0;
        let lo = _mulx_u64(a, b, &mut hi);
        /*
result = 922337203685477580000 =
0b00110001_1111111111111111_1111111111111111_1111111111111111_1111110011100000
  ^~hi~~~~ ^~lo~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        */
        assert_eq!(
            lo,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111100_11100000u64
        );
        assert_eq!(hi, 0b00110001u64);
    }
}
