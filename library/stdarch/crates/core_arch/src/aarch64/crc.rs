extern "unadjusted" {
    #[link_name = "llvm.aarch64.crc32x"]
    fn crc32x_(crc: u32, data: u64) -> u32;

    #[link_name = "llvm.aarch64.crc32cx"]
    fn crc32cx_(crc: u32, data: u64) -> u32;
}

#[cfg(test)]
use stdarch_test::assert_instr;

/// CRC32 single round checksum for quad words (64 bits).
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(test, assert_instr(crc32x))]
pub unsafe fn __crc32d(crc: u32, data: u64) -> u32 {
    crc32x_(crc, data)
}

/// CRC32-C single round checksum for quad words (64 bits).
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(test, assert_instr(crc32cx))]
pub unsafe fn __crc32cd(crc: u32, data: u64) -> u32 {
    crc32cx_(crc, data)
}

#[cfg(test)]
mod tests {
    use crate::core_arch::{aarch64::*, simd::*};
    use std::mem;
    use stdarch_test::simd_test;

    #[simd_test(enable = "crc")]
    unsafe fn test_crc32d() {
        assert_eq!(__crc32d(0, 0), 0);
        assert_eq!(__crc32d(0, 18446744073709551615), 1147535477);
    }

    #[simd_test(enable = "crc")]
    unsafe fn test_crc32cd() {
        assert_eq!(__crc32cd(0, 0), 0);
        assert_eq!(__crc32cd(0, 18446744073709551615), 3293575501);
    }
}
