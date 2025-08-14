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
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mulx_u32)
#[inline]
// LLVM BUG (should be mulxl): https://bugs.llvm.org/show_bug.cgi?id=34232
#[cfg_attr(all(test, target_arch = "x86_64"), assert_instr(imul))]
#[cfg_attr(all(test, target_arch = "x86"), assert_instr(mul))]
#[target_feature(enable = "bmi2")]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mulx_u32(a: u32, b: u32, hi: &mut u32) -> u32 {
    let result: u64 = (a as u64) * (b as u64);
    *hi = (result >> 32) as u32;
    result as u32
}

/// Zeroes higher bits of `a` >= `index`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_bzhi_u32)
#[inline]
#[target_feature(enable = "bmi2")]
#[cfg_attr(test, assert_instr(bzhi))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _bzhi_u32(a: u32, index: u32) -> u32 {
    unsafe { x86_bmi2_bzhi_32(a, index) }
}

/// Scatter contiguous low order bits of `a` to the result at the positions
/// specified by the `mask`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_pdep_u32)
#[inline]
#[target_feature(enable = "bmi2")]
#[cfg_attr(test, assert_instr(pdep))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _pdep_u32(a: u32, mask: u32) -> u32 {
    unsafe { x86_bmi2_pdep_32(a, mask) }
}

/// Gathers the bits of `x` specified by the `mask` into the contiguous low
/// order bit positions of the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_pext_u32)
#[inline]
#[target_feature(enable = "bmi2")]
#[cfg_attr(test, assert_instr(pext))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _pext_u32(a: u32, mask: u32) -> u32 {
    unsafe { x86_bmi2_pext_32(a, mask) }
}

unsafe extern "C" {
    #[link_name = "llvm.x86.bmi.bzhi.32"]
    fn x86_bmi2_bzhi_32(x: u32, y: u32) -> u32;
    #[link_name = "llvm.x86.bmi.pdep.32"]
    fn x86_bmi2_pdep_32(x: u32, y: u32) -> u32;
    #[link_name = "llvm.x86.bmi.pext.32"]
    fn x86_bmi2_pext_32(x: u32, y: u32) -> u32;
}

#[cfg(test)]
mod tests {
    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "bmi2")]
    unsafe fn test_pext_u32() {
        let n = 0b1011_1110_1001_0011u32;

        let m0 = 0b0110_0011_1000_0101u32;
        let s0 = 0b0000_0000_0011_0101u32;

        let m1 = 0b1110_1011_1110_1111u32;
        let s1 = 0b0001_0111_0100_0011u32;

        assert_eq!(_pext_u32(n, m0), s0);
        assert_eq!(_pext_u32(n, m1), s1);
    }

    #[simd_test(enable = "bmi2")]
    unsafe fn test_pdep_u32() {
        let n = 0b1011_1110_1001_0011u32;

        let m0 = 0b0110_0011_1000_0101u32;
        let s0 = 0b0000_0010_0000_0101u32;

        let m1 = 0b1110_1011_1110_1111u32;
        let s1 = 0b1110_1001_0010_0011u32;

        assert_eq!(_pdep_u32(n, m0), s0);
        assert_eq!(_pdep_u32(n, m1), s1);
    }

    #[simd_test(enable = "bmi2")]
    unsafe fn test_bzhi_u32() {
        let n = 0b1111_0010u32;
        let s = 0b0001_0010u32;
        assert_eq!(_bzhi_u32(n, 5), s);
    }

    #[simd_test(enable = "bmi2")]
    unsafe fn test_mulx_u32() {
        let a: u32 = 4_294_967_200;
        let b: u32 = 2;
        let mut hi = 0;
        let lo = _mulx_u32(a, b, &mut hi);
        /*
        result = 8589934400
               = 0b0001_1111_1111_1111_1111_1111_1111_0100_0000u64
                   ^~hi ^~lo~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                */
        assert_eq!(lo, 0b1111_1111_1111_1111_1111_1111_0100_0000u32);
        assert_eq!(hi, 0b0001u32);
    }
}
