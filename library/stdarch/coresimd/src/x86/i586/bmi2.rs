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
use stdsimd_test::assert_instr;

/// Unsigned multiply without affecting flags.
///
/// Unsigned multiplication of `a` with `b` returning a pair `(lo, hi)` with
/// the low half and the high half of the result.
#[inline(always)]
// LLVM BUG (should be mulxl): https://bugs.llvm.org/show_bug.cgi?id=34232
#[cfg_attr(all(test, target_arch = "x86_64"), assert_instr(imul))]
#[cfg_attr(all(test, target_arch = "x86"), assert_instr(mulx))]
#[target_feature = "+bmi2"]
pub unsafe fn _mulx_u32(a: u32, b: u32) -> (u32, u32) {
    let result: u64 = (a as u64) * (b as u64);
    let hi = (result >> 32) as u32;
    (result as u32, hi)
}

/// Unsigned multiply without affecting flags.
///
/// Unsigned multiplication of `a` with `b` returning a pair `(lo, hi)` with
/// the low half and the high half of the result.
#[inline(always)]
#[cfg_attr(test, assert_instr(mulx))]
#[target_feature = "+bmi2"]
#[cfg(not(target_arch = "x86"))] // calls an intrinsic
pub unsafe fn _mulx_u64(a: u64, b: u64) -> (u64, u64) {
    let result: u128 = (a as u128) * (b as u128);
    let hi = (result >> 64) as u64;
    (result as u64, hi)
}

/// Zero higher bits of `a` >= `index`.
#[inline(always)]
#[target_feature = "+bmi2"]
#[cfg_attr(test, assert_instr(bzhi))]
pub unsafe fn _bzhi_u32(a: u32, index: u32) -> u32 {
    x86_bmi2_bzhi_32(a, index)
}

/// Zero higher bits of `a` >= `index`.
#[inline(always)]
#[target_feature = "+bmi2"]
#[cfg_attr(test, assert_instr(bzhi))]
#[cfg(not(target_arch = "x86"))]
pub unsafe fn _bzhi_u64(a: u64, index: u64) -> u64 {
    x86_bmi2_bzhi_64(a, index)
}

/// Scatter contiguous low order bits of `a` to the result at the positions
/// specified by the `mask`.
#[inline(always)]
#[target_feature = "+bmi2"]
#[cfg_attr(test, assert_instr(pdep))]
pub unsafe fn _pdep_u32(a: u32, mask: u32) -> u32 {
    x86_bmi2_pdep_32(a, mask)
}

/// Scatter contiguous low order bits of `a` to the result at the positions
/// specified by the `mask`.
#[inline(always)]
#[target_feature = "+bmi2"]
#[cfg_attr(test, assert_instr(pdep))]
#[cfg(not(target_arch = "x86"))]
pub unsafe fn _pdep_u64(a: u64, mask: u64) -> u64 {
    x86_bmi2_pdep_64(a, mask)
}

/// Gathers the bits of `x` specified by the `mask` into the contiguous low
/// order bit positions of the result.
#[inline(always)]
#[target_feature = "+bmi2"]
#[cfg_attr(test, assert_instr(pext))]
pub unsafe fn _pext_u32(a: u32, mask: u32) -> u32 {
    x86_bmi2_pext_32(a, mask)
}

/// Gathers the bits of `x` specified by the `mask` into the contiguous low
/// order bit positions of the result.
#[inline(always)]
#[target_feature = "+bmi2"]
#[cfg_attr(test, assert_instr(pext))]
#[cfg(not(target_arch = "x86"))]
pub unsafe fn _pext_u64(a: u64, mask: u64) -> u64 {
    x86_bmi2_pext_64(a, mask)
}

#[allow(dead_code)]
extern "C" {
    #[link_name = "llvm.x86.bmi.bzhi.32"]
    fn x86_bmi2_bzhi_32(x: u32, y: u32) -> u32;
    #[link_name = "llvm.x86.bmi.bzhi.64"]
    fn x86_bmi2_bzhi_64(x: u64, y: u64) -> u64;
    #[link_name = "llvm.x86.bmi.pdep.32"]
    fn x86_bmi2_pdep_32(x: u32, y: u32) -> u32;
    #[link_name = "llvm.x86.bmi.pdep.64"]
    fn x86_bmi2_pdep_64(x: u64, y: u64) -> u64;
    #[link_name = "llvm.x86.bmi.pext.32"]
    fn x86_bmi2_pext_32(x: u32, y: u32) -> u32;
    #[link_name = "llvm.x86.bmi.pext.64"]
    fn x86_bmi2_pext_64(x: u64, y: u64) -> u64;
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use x86::i586::bmi2;

    #[simd_test = "bmi2"]
    unsafe fn _pext_u32() {
        let n = 0b1011_1110_1001_0011u32;

        let m0 = 0b0110_0011_1000_0101u32;
        let s0 = 0b0000_0000_0011_0101u32;

        let m1 = 0b1110_1011_1110_1111u32;
        let s1 = 0b0001_0111_0100_0011u32;

        assert_eq!(bmi2::_pext_u32(n, m0), s0);
        assert_eq!(bmi2::_pext_u32(n, m1), s1);
    }

    #[simd_test = "bmi2"]
    #[cfg(not(target_arch = "x86"))]
    unsafe fn _pext_u64() {
        let n = 0b1011_1110_1001_0011u64;

        let m0 = 0b0110_0011_1000_0101u64;
        let s0 = 0b0000_0000_0011_0101u64;

        let m1 = 0b1110_1011_1110_1111u64;
        let s1 = 0b0001_0111_0100_0011u64;

        assert_eq!(bmi2::_pext_u64(n, m0), s0);
        assert_eq!(bmi2::_pext_u64(n, m1), s1);
    }

    #[simd_test = "bmi2"]
    unsafe fn _pdep_u32() {
        let n = 0b1011_1110_1001_0011u32;

        let m0 = 0b0110_0011_1000_0101u32;
        let s0 = 0b0000_0010_0000_0101u32;

        let m1 = 0b1110_1011_1110_1111u32;
        let s1 = 0b1110_1001_0010_0011u32;

        assert_eq!(bmi2::_pdep_u32(n, m0), s0);
        assert_eq!(bmi2::_pdep_u32(n, m1), s1);
    }

    #[simd_test = "bmi2"]
    #[cfg(not(target_arch = "x86"))]
    unsafe fn _pdep_u64() {
        let n = 0b1011_1110_1001_0011u64;

        let m0 = 0b0110_0011_1000_0101u64;
        let s0 = 0b0000_0010_0000_0101u64;

        let m1 = 0b1110_1011_1110_1111u64;
        let s1 = 0b1110_1001_0010_0011u64;

        assert_eq!(bmi2::_pdep_u64(n, m0), s0);
        assert_eq!(bmi2::_pdep_u64(n, m1), s1);
    }

    #[simd_test = "bmi2"]
    unsafe fn _bzhi_u32() {
        let n = 0b1111_0010u32;
        let s = 0b0001_0010u32;
        assert_eq!(bmi2::_bzhi_u32(n, 5), s);
    }

    #[simd_test = "bmi2"]
    #[cfg(not(target_arch = "x86"))]
    unsafe fn _bzhi_u64() {
        let n = 0b1111_0010u64;
        let s = 0b0001_0010u64;
        assert_eq!(bmi2::_bzhi_u64(n, 5), s);
    }

    #[simd_test = "bmi2"]
    unsafe fn _mulx_u32() {
        let a: u32 = 4_294_967_200;
        let b: u32 = 2;
        let (lo, hi): (u32, u32) = bmi2::_mulx_u32(a, b);
        /*
result = 8589934400
       = 0b0001_1111_1111_1111_1111_1111_1111_0100_0000u64
           ^~hi ^~lo~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        */
        assert_eq!(lo, 0b1111_1111_1111_1111_1111_1111_0100_0000u32);
        assert_eq!(hi, 0b0001u32);
    }

    #[simd_test = "bmi2"]
    #[cfg(not(target_arch = "x86"))]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    unsafe fn _mulx_u64() {
        let a: u64 = 9_223_372_036_854_775_800;
        let b: u64 = 100;
        let (lo, hi): (u64, u64) = bmi2::_mulx_u64(a, b);
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
