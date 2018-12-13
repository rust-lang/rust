extern "C" {
    #[link_name = "llvm.aarch64.crc32b"]
    fn crc32b_(crc: u32, data: u32) -> u32;
    #[link_name = "llvm.aarch64.crc32h"]
    fn crc32h_(crc: u32, data: u32) -> u32;
    #[link_name = "llvm.aarch64.crc32w"]
    fn crc32w_(crc: u32, data: u32) -> u32;
    #[link_name = "llvm.aarch64.crc32x"]
    fn crc32x_(crc: u32, data: u64) -> u32;

    #[link_name = "llvm.aarch64.crc32cb"]
    fn crc32cb_(crc: u32, data: u32) -> u32;
    #[link_name = "llvm.aarch64.crc32ch"]
    fn crc32ch_(crc: u32, data: u32) -> u32;
    #[link_name = "llvm.aarch64.crc32cw"]
    fn crc32cw_(crc: u32, data: u32) -> u32;
    #[link_name = "llvm.aarch64.crc32cx"]
    fn crc32cx_(crc: u32, data: u64) -> u32;
}

#[cfg(test)]
use stdsimd_test::assert_instr;

/// CRC32 single round checksum for bytes (8 bits).
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(test, assert_instr(crc32b))]
pub unsafe fn crc32b(crc: u32, data: u8) -> u32 {
    crc32b_(crc, data as u32)
}

/// CRC32 single round checksum for half words (16 bits).
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(test, assert_instr(crc32h))]
pub unsafe fn crc32h(crc: u32, data: u16) -> u32 {
    crc32h_(crc, data as u32)
}

/// CRC32 single round checksum for words (32 bits).
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(test, assert_instr(crc32w))]
pub unsafe fn crc32w(crc: u32, data: u32) -> u32 {
    crc32w_(crc, data)
}

/// CRC32 single round checksum for quad words (64 bits).
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(test, assert_instr(crc32x))]
pub unsafe fn crc32x(crc: u32, data: u64) -> u32 {
    crc32x_(crc, data)
}

/// CRC32-C single round checksum for bytes (8 bits).
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(test, assert_instr(crc32cb))]
pub unsafe fn crc32cb(crc: u32, data: u8) -> u32 {
    crc32cb_(crc, data as u32)
}

/// CRC32-C single round checksum for half words (16 bits).
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(test, assert_instr(crc32ch))]
pub unsafe fn crc32ch(crc: u32, data: u16) -> u32 {
    crc32ch_(crc, data as u32)
}

/// CRC32-C single round checksum for words (32 bits).
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(test, assert_instr(crc32cw))]
pub unsafe fn crc32cw(crc: u32, data: u32) -> u32 {
    crc32cw_(crc, data)
}

/// CRC32-C single round checksum for quad words (64 bits).
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(test, assert_instr(crc32cx))]
pub unsafe fn crc32cx(crc: u32, data: u64) -> u32 {
    crc32cx_(crc, data)
}

#[cfg(test)]
mod tests {
    use coresimd::aarch64::*;
    use coresimd::simd::*;
    use std::mem;
    use stdsimd_test::simd_test;

    #[simd_test(enable = "crc")]
    unsafe fn test_crc32b() {
        assert_eq!(crc32b(0, 0), 0);
        assert_eq!(crc32b(0, 255), 755167117);
    }

    #[simd_test(enable = "crc")]
    unsafe fn test_crc32h() {
        assert_eq!(crc32h(0, 0), 0);
        assert_eq!(crc32h(0, 16384), 1994146192);
    }

    #[simd_test(enable = "crc")]
    unsafe fn test_crc32w() {
        assert_eq!(crc32w(0, 0), 0);
        assert_eq!(crc32w(0, 4294967295), 3736805603);
    }

    #[simd_test(enable = "crc")]
    unsafe fn test_crc32x() {
        assert_eq!(crc32x(0, 0), 0);
        assert_eq!(crc32x(0, 18446744073709551615), 1147535477);
    }

    #[simd_test(enable = "crc")]
    unsafe fn test_crc32cb() {
        assert_eq!(crc32cb(0, 0), 0);
        assert_eq!(crc32cb(0, 255), 2910671697);
    }

    #[simd_test(enable = "crc")]
    unsafe fn test_crc32ch() {
        assert_eq!(crc32ch(0, 0), 0);
        assert_eq!(crc32ch(0, 16384), 1098587580);
    }

    #[simd_test(enable = "crc")]
    unsafe fn test_crc32cw() {
        assert_eq!(crc32cw(0, 0), 0);
        assert_eq!(crc32cw(0, 4294967295), 3080238136);
    }

    #[simd_test(enable = "crc")]
    unsafe fn test_crc32cx() {
        assert_eq!(crc32cx(0, 0), 0);
        assert_eq!(crc32cx(0, 18446744073709551615), 3293575501);
    }

}
