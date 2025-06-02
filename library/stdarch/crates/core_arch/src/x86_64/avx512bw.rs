use crate::core_arch::x86::*;

/// Convert 64-bit mask a into an integer value, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_cvtmask64_u64)
#[inline]
#[target_feature(enable = "avx512bw")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub fn _cvtmask64_u64(a: __mmask64) -> u64 {
    a
}

/// Convert integer value a into an 64-bit mask, and store the result in k.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_cvtu64_mask64)
#[inline]
#[target_feature(enable = "avx512bw")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub fn _cvtu64_mask64(a: u64) -> __mmask64 {
    a
}

#[cfg(test)]
mod tests {

    use stdarch_test::simd_test;

    use crate::core_arch::{x86::*, x86_64::*};

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_cvtmask64_u64() {
        let a: __mmask64 = 0b11001100_00110011_01100110_10011001;
        let r = _cvtmask64_u64(a);
        let e: u64 = 0b11001100_00110011_01100110_10011001;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_cvtu64_mask64() {
        let a: u64 = 0b11001100_00110011_01100110_10011001;
        let r = _cvtu64_mask64(a);
        let e: __mmask64 = 0b11001100_00110011_01100110_10011001;
        assert_eq!(r, e);
    }
}
