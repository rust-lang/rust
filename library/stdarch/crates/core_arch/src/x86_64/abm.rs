//! Advanced Bit Manipulation (ABM) instructions
//!
//! The POPCNT and LZCNT have their own CPUID bits to indicate support.
//!
//! The references are:
//!
//! - [Intel 64 and IA-32 Architectures Software Developer's Manual Volume 2:
//!   Instruction Set Reference, A-Z][intel64_ref].
//! - [AMD64 Architecture Programmer's Manual, Volume 3: General-Purpose and
//!   System Instructions][amd64_ref].
//!
//! [Wikipedia][wikipedia_bmi] provides a quick overview of the instructions
//! available.
//!
//! [intel64_ref]: https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf
//! [amd64_ref]: https://docs.amd.com/v/u/en-US/24594_3.37
//! [wikipedia_bmi]:
//! https://en.wikipedia.org/wiki/Bit_Manipulation_Instruction_Sets#ABM_.28Advanced_Bit_Manipulation.29

#[cfg(test)]
use stdarch_test::assert_instr;

/// Counts the leading most significant zero bits.
///
/// When the operand is zero, it returns its size in bits.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_lzcnt_u64)
#[inline]
#[target_feature(enable = "lzcnt")]
#[cfg_attr(test, assert_instr(lzcnt))]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[rustc_const_unstable(feature = "stdarch_const_x86", issue = "149298")]
pub const fn _lzcnt_u64(x: u64) -> u64 {
    x.leading_zeros() as u64
}

/// Counts the bits that are set.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_popcnt64)
#[inline]
#[target_feature(enable = "popcnt")]
#[cfg_attr(test, assert_instr(popcnt))]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[rustc_const_unstable(feature = "stdarch_const_x86", issue = "149298")]
pub const fn _popcnt64(x: i64) -> i32 {
    x.count_ones() as i32
}

#[cfg(test)]
mod tests {
    use crate::core_arch::assert_eq_const as assert_eq;
    use stdarch_test::simd_test;

    use crate::core_arch::arch::x86_64::*;

    #[simd_test(enable = "lzcnt")]
    const unsafe fn test_lzcnt_u64() {
        assert_eq!(_lzcnt_u64(0b0101_1010), 57);
    }

    #[simd_test(enable = "popcnt")]
    const unsafe fn test_popcnt64() {
        assert_eq!(_popcnt64(0b0101_1010), 4);
    }
}
