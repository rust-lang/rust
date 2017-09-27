//! Advanced Bit Manipulation (ABM) instructions
//!
//! The POPCNT and LZCNT have their own CPUID bits to indicate support.
//!
//! The references are:
//!
//! - [Intel 64 and IA-32 Architectures Software Developer's Manual Volume 2: Instruction Set Reference, A-Z](http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf).
//! - [AMD64 Architecture Programmer's Manual, Volume 3: General-Purpose and System Instructions](http://support.amd.com/TechDocs/24594.pdf).
//!
//! [Wikipedia](https://en.wikipedia.org/wiki/Bit_Manipulation_Instruction_Sets#ABM_.28Advanced_Bit_Manipulation.29)
//! provides a quick overview of the instructions available.

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Counts the leading most significant zero bits.
///
/// When the operand is zero, it returns its size in bits.
#[inline(always)]
#[target_feature = "+lzcnt"]
#[cfg_attr(test, assert_instr(lzcnt))]
pub unsafe fn _lzcnt_u32(x: u32) -> u32 { x.leading_zeros() }

/// Counts the leading most significant zero bits.
///
/// When the operand is zero, it returns its size in bits.
#[inline(always)]
#[target_feature = "+lzcnt"]
#[cfg_attr(test, assert_instr(lzcnt))]
pub unsafe fn _lzcnt_u64(x: u64) -> u64 { x.leading_zeros() as u64 }

/// Counts the bits that are set.
#[inline(always)]
#[target_feature = "+popcnt"]
#[cfg_attr(test, assert_instr(popcnt))]
pub unsafe fn _popcnt32(x: u32) -> u32 { x.count_ones() }

/// Counts the bits that are set.
#[inline(always)]
#[target_feature = "+popcnt"]
#[cfg_attr(test, assert_instr(popcnt))]
pub unsafe fn _popcnt64(x: u64) -> u64 { x.count_ones() as u64 }

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use x86::abm;

    #[simd_test = "lzcnt"]
    fn _lzcnt_u32() {
        assert_eq!(unsafe { abm::_lzcnt_u32(0b0101_1010u32) }, 25u32);
    }

    #[simd_test = "lzcnt"]
    fn _lzcnt_u64() {
        assert_eq!(unsafe { abm::_lzcnt_u64(0b0101_1010u64) }, 57u64);
    }

    #[simd_test = "popcnt"]
    fn _popcnt32() {
        assert_eq!(unsafe { abm::_popcnt32(0b0101_1010u32) }, 4);
    }

    #[simd_test = "popcnt"]
    fn _popcnt64() {
        assert_eq!(unsafe { abm::_popcnt64(0b0101_1010u64) }, 4);
    }
}
