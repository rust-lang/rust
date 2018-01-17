//! Advanced Bit Manipulation (ABM) instructions
//!
//! The POPCNT and LZCNT have their own CPUID bits to indicate support.
//!
//! The references are:
//!
//! - [Intel 64 and IA-32 Architectures Software Developer's Manual Volume 2:
//! Instruction Set Reference, A-Z][intel64_ref].
//! - [AMD64 Architecture Programmer's Manual, Volume 3: General-Purpose and
//! System Instructions][amd64_ref].
//!
//! [Wikipedia][wikipedia_bmi] provides a quick overview of the instructions
//! available.
//!
//! [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf
//! [amd64_ref]: http://support.amd.com/TechDocs/24594.pdf
//! [wikipedia_bmi]:
//! https://en.wikipedia.org/wiki/Bit_Manipulation_Instruction_Sets#ABM_.28Advanced_Bit_Manipulation.29

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Counts the leading most significant zero bits.
///
/// When the operand is zero, it returns its size in bits.
#[inline(always)]
#[target_feature(enable = "lzcnt")]
#[cfg_attr(test, assert_instr(lzcnt))]
pub unsafe fn _lzcnt_u32(x: u32) -> u32 {
    x.leading_zeros()
}

/// Counts the leading most significant zero bits.
///
/// When the operand is zero, it returns its size in bits.
#[inline(always)]
#[target_feature(enable = "lzcnt")]
#[cfg_attr(test, assert_instr(lzcnt))]
pub unsafe fn _lzcnt_u64(x: u64) -> u64 {
    x.leading_zeros() as u64
}

/// Counts the bits that are set.
#[inline(always)]
#[target_feature(enable = "popcnt")]
#[cfg_attr(test, assert_instr(popcnt))]
pub unsafe fn _popcnt32(x: i32) -> i32 {
    x.count_ones() as i32
}

/// Counts the bits that are set.
#[inline(always)]
#[target_feature(enable = "popcnt")]
#[cfg_attr(test, assert_instr(popcnt))]
pub unsafe fn _popcnt64(x: i64) -> i32 {
    x.count_ones() as i32
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use x86::i586::abm;

    #[simd_test = "lzcnt"]
    unsafe fn _lzcnt_u32() {
        assert_eq!(abm::_lzcnt_u32(0b0101_1010), 25);
    }

    #[simd_test = "lzcnt"]
    unsafe fn _lzcnt_u64() {
        assert_eq!(abm::_lzcnt_u64(0b0101_1010), 57);
    }

    #[simd_test = "popcnt"]
    unsafe fn _popcnt32() {
        assert_eq!(abm::_popcnt32(0b0101_1010), 4);
    }

    #[simd_test = "popcnt"]
    unsafe fn _popcnt64() {
        assert_eq!(abm::_popcnt64(0b0101_1010), 4);
    }
}
