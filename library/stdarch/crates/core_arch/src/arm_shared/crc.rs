extern "unadjusted" {
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crc32b")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.crc32b")]
    fn crc32b_(crc: u32, data: u32) -> u32;
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crc32h")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.crc32h")]
    fn crc32h_(crc: u32, data: u32) -> u32;
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crc32w")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.crc32w")]
    fn crc32w_(crc: u32, data: u32) -> u32;

    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crc32cb")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.crc32cb")]
    fn crc32cb_(crc: u32, data: u32) -> u32;
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crc32ch")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.crc32ch")]
    fn crc32ch_(crc: u32, data: u32) -> u32;
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crc32cw")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.crc32cw")]
    fn crc32cw_(crc: u32, data: u32) -> u32;
}

#[cfg(test)]
use stdarch_test::assert_instr;

/// CRC32 single round checksum for bytes (8 bits).
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
#[cfg_attr(test, assert_instr(crc32b))]
pub unsafe fn __crc32b(crc: u32, data: u8) -> u32 {
    crc32b_(crc, data as u32)
}

/// CRC32 single round checksum for half words (16 bits).
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
#[cfg_attr(test, assert_instr(crc32h))]
pub unsafe fn __crc32h(crc: u32, data: u16) -> u32 {
    crc32h_(crc, data as u32)
}

/// CRC32 single round checksum for words (32 bits).
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
#[cfg_attr(test, assert_instr(crc32w))]
pub unsafe fn __crc32w(crc: u32, data: u32) -> u32 {
    crc32w_(crc, data)
}

/// CRC32-C single round checksum for bytes (8 bits).
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
#[cfg_attr(test, assert_instr(crc32cb))]
pub unsafe fn __crc32cb(crc: u32, data: u8) -> u32 {
    crc32cb_(crc, data as u32)
}

/// CRC32-C single round checksum for half words (16 bits).
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
#[cfg_attr(test, assert_instr(crc32ch))]
pub unsafe fn __crc32ch(crc: u32, data: u16) -> u32 {
    crc32ch_(crc, data as u32)
}

/// CRC32-C single round checksum for words (32 bits).
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
#[cfg_attr(test, assert_instr(crc32cw))]
pub unsafe fn __crc32cw(crc: u32, data: u32) -> u32 {
    crc32cw_(crc, data)
}

#[cfg(test)]
mod tests {
    use crate::core_arch::{arm_shared::*, simd::*};
    use std::mem;
    use stdarch_test::simd_test;

    #[simd_test(enable = "crc")]
    unsafe fn test_crc32b() {
        assert_eq!(__crc32b(0, 0), 0);
        assert_eq!(__crc32b(0, 255), 755167117);
    }

    #[simd_test(enable = "crc")]
    unsafe fn test_crc32h() {
        assert_eq!(__crc32h(0, 0), 0);
        assert_eq!(__crc32h(0, 16384), 1994146192);
    }

    #[simd_test(enable = "crc")]
    unsafe fn test_crc32w() {
        assert_eq!(__crc32w(0, 0), 0);
        assert_eq!(__crc32w(0, 4294967295), 3736805603);
    }

    #[simd_test(enable = "crc")]
    unsafe fn test_crc32cb() {
        assert_eq!(__crc32cb(0, 0), 0);
        assert_eq!(__crc32cb(0, 255), 2910671697);
    }

    #[simd_test(enable = "crc")]
    unsafe fn test_crc32ch() {
        assert_eq!(__crc32ch(0, 0), 0);
        assert_eq!(__crc32ch(0, 16384), 1098587580);
    }

    #[simd_test(enable = "crc")]
    unsafe fn test_crc32cw() {
        assert_eq!(__crc32cw(0, 0), 0);
        assert_eq!(__crc32cw(0, 4294967295), 3080238136);
    }
}
