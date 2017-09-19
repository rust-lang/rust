//! Advanced Bit Manipulation (ABM) instructions
//! 
//! That is, POPCNT and LZCNT. These instructions have their own CPUID bits to
//! indicate support.
//!
//! TODO: it is unclear which target feature to use here. SSE4.2 should be good
//! enough but we might need to use BMI for LZCNT if there are any problems.
//!
//! The references are:
//!
//! - [Intel 64 and IA-32 Architectures Software Developer's Manual Volume 2: Instruction Set Reference, A-Z](http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf).
//! - [AMD64 Architecture Programmer's Manual, Volume 3: General-Purpose and System Instructions](http://support.amd.com/TechDocs/24594.pdf).
//!
//! [Wikipedia](https://en.wikipedia.org/wiki/Bit_Manipulation_Instruction_Sets#ABM_.28Advanced_Bit_Manipulation.29)
//! provides a quick overview of the instructions available.


/// Counts the leading most significant zero bits.
///
/// When the operand is zero, it returns its size in bits.
#[inline(always)]
#[target_feature = "+sse4.2"]
pub fn _lzcnt_u32(x: u32) -> u32 { x.leading_zeros() }

/// Counts the leading most significant zero bits.
///
/// When the operand is zero, it returns its size in bits.
#[inline(always)]
#[target_feature = "+sse4.2"]
pub fn _lzcnt_u64(x: u64) -> u64 { x.leading_zeros() as u64 }

/// Counts the bits that are set.
#[inline(always)]
#[target_feature = "+sse4.2"]
pub fn _popcnt32(x: u32) -> u32 { x.count_ones() }

/// Counts the bits that are set.
#[inline(always)]
#[target_feature = "+sse4.2"]
pub fn _popcnt64(x: u64) -> u64 { x.count_ones() as u64 }

#[cfg(all(test, target_feature = "sse4.2", any(target_arch = "x86", target_arch = "x86_64")))]
mod tests {
    use x86::abm;

    #[test]
    #[target_feature = "+sse4.2"]
    fn _lzcnt_u32() {
        assert_eq!(abm::_lzcnt_u32(0b0101_1010u32), 25u32);
    }

    #[test]
    #[target_feature = "+sse4.2"]
    fn _lzcnt_u64() {
        assert_eq!(abm::_lzcnt_u64(0b0101_1010u64), 57u64);
    }

    #[test]
    #[target_feature = "+sse4.2"]
    fn _popcnt32() {
        assert_eq!(abm::_popcnt32(0b0101_1010u32), 4);
    }

    #[test]
    #[target_feature = "+sse4.2"]
    fn _popcnt64() {
        assert_eq!(abm::_popcnt64(0b0101_1010u64), 4);
    }
}
