extern "unadjusted" {
    #[cfg_attr(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        link_name = "llvm.aarch64.crc32b"
    )]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.crc32b")]
    fn crc32b_(crc: u32, data: u32) -> u32;
    #[cfg_attr(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        link_name = "llvm.aarch64.crc32h"
    )]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.crc32h")]
    fn crc32h_(crc: u32, data: u32) -> u32;
    #[cfg_attr(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        link_name = "llvm.aarch64.crc32w"
    )]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.crc32w")]
    fn crc32w_(crc: u32, data: u32) -> u32;

    #[cfg_attr(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        link_name = "llvm.aarch64.crc32cb"
    )]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.crc32cb")]
    fn crc32cb_(crc: u32, data: u32) -> u32;
    #[cfg_attr(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        link_name = "llvm.aarch64.crc32ch"
    )]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.crc32ch")]
    fn crc32ch_(crc: u32, data: u32) -> u32;
    #[cfg_attr(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        link_name = "llvm.aarch64.crc32cw"
    )]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.crc32cw")]
    fn crc32cw_(crc: u32, data: u32) -> u32;
    #[cfg_attr(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        link_name = "llvm.aarch64.crc32x"
    )]
    fn crc32x_(crc: u32, data: u64) -> u32;
    #[cfg_attr(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        link_name = "llvm.aarch64.crc32cx"
    )]
    fn crc32cx_(crc: u32, data: u64) -> u32;
}

#[cfg(test)]
use stdarch_test::assert_instr;

/// CRC32 single round checksum for bytes (8 bits).
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/__crc32b)
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
#[cfg_attr(test, assert_instr(crc32b))]
#[unstable(feature = "stdarch_arm_crc32", issue = "117215")]
pub unsafe fn __crc32b(crc: u32, data: u8) -> u32 {
    crc32b_(crc, data as u32)
}

/// CRC32 single round checksum for half words (16 bits).
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/__crc32h)
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
#[cfg_attr(test, assert_instr(crc32h))]
#[unstable(feature = "stdarch_arm_crc32", issue = "117215")]
pub unsafe fn __crc32h(crc: u32, data: u16) -> u32 {
    crc32h_(crc, data as u32)
}

/// CRC32 single round checksum for words (32 bits).
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/__crc32w)
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
#[cfg_attr(test, assert_instr(crc32w))]
#[unstable(feature = "stdarch_arm_crc32", issue = "117215")]
pub unsafe fn __crc32w(crc: u32, data: u32) -> u32 {
    crc32w_(crc, data)
}

/// CRC32-C single round checksum for bytes (8 bits).
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/__crc32cb)
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
#[cfg_attr(test, assert_instr(crc32cb))]
#[unstable(feature = "stdarch_arm_crc32", issue = "117215")]
pub unsafe fn __crc32cb(crc: u32, data: u8) -> u32 {
    crc32cb_(crc, data as u32)
}

/// CRC32-C single round checksum for half words (16 bits).
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/__crc32ch)
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
#[cfg_attr(test, assert_instr(crc32ch))]
#[unstable(feature = "stdarch_arm_crc32", issue = "117215")]
pub unsafe fn __crc32ch(crc: u32, data: u16) -> u32 {
    crc32ch_(crc, data as u32)
}

/// CRC32-C single round checksum for words (32 bits).
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/__crc32cw)
#[inline]
#[target_feature(enable = "crc")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
#[cfg_attr(test, assert_instr(crc32cw))]
#[unstable(feature = "stdarch_arm_crc32", issue = "117215")]
pub unsafe fn __crc32cw(crc: u32, data: u32) -> u32 {
    crc32cw_(crc, data)
}

/// CRC32 single round checksum for quad words (64 bits).
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/__crc32d)
#[inline]
#[target_feature(enable = "crc")]
#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
#[cfg_attr(test, assert_instr(crc32x))]
#[unstable(feature = "stdarch_arm_crc32", issue = "117215")]
pub unsafe fn __crc32d(crc: u32, data: u64) -> u32 {
    crc32x_(crc, data)
}

/// CRC32 single round checksum for quad words (64 bits).
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/__crc32d)
#[inline]
#[target_feature(enable = "crc")]
#[cfg(target_arch = "arm")]
#[cfg_attr(test, assert_instr(crc32w))]
#[unstable(feature = "stdarch_arm_crc32", issue = "117215")]
pub unsafe fn __crc32d(crc: u32, data: u64) -> u32 {
    // On 32-bit ARM this intrinsic emits a chain of two `crc32_w` instructions
    // and truncates the data to 32 bits in both clang and gcc
    crc32w_(
        crc32w_(crc, (data & 0xffffffff) as u32),
        (data >> 32) as u32,
    )
}

/// CRC32 single round checksum for quad words (64 bits).
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/__crc32cd)
#[inline]
#[target_feature(enable = "crc")]
#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
#[cfg_attr(test, assert_instr(crc32cx))]
#[unstable(feature = "stdarch_arm_crc32", issue = "117215")]
pub unsafe fn __crc32cd(crc: u32, data: u64) -> u32 {
    crc32cx_(crc, data)
}

/// CRC32 single round checksum for quad words (64 bits).
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/__crc32cd)
#[inline]
#[target_feature(enable = "crc")]
#[cfg(target_arch = "arm")]
#[cfg_attr(test, assert_instr(crc32cw))]
#[unstable(feature = "stdarch_arm_crc32", issue = "117215")]
pub unsafe fn __crc32cd(crc: u32, data: u64) -> u32 {
    // On 32-bit ARM this intrinsic emits a chain of two `crc32_cw` instructions
    // and truncates the data to 32 bits in both clang and gcc
    crc32cw_(
        crc32cw_(crc, (data & 0xffffffff) as u32),
        (data >> 32) as u32,
    )
}

#[cfg(test)]
mod tests {
    use crate::core_arch::{arm_shared::*, simd::*};
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
