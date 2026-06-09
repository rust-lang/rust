// We're testing aarch64 CRC32 target specific features
//@only-target: aarch64
//@compile-flags: -C target-feature=+crc

use std::arch::aarch64::*;
use std::arch::is_aarch64_feature_detected;

fn main() {
    assert!(is_aarch64_feature_detected!("crc"));

    unsafe {
        test_crc32_standard();
        test_crc32c_castagnoli();
    }
}

#[target_feature(enable = "crc")]
unsafe fn test_crc32_standard() {
    // __crc32b: 8-bit input
    assert_eq!(__crc32b(0x00000000, 0x01), 0x77073096);
    assert_eq!(__crc32b(0xffffffff, 0x61), 0x174841bc);
    assert_eq!(__crc32b(0x2aa1e72b, 0x2a), 0x772d9171);

    // __crc32h: 16-bit input
    assert_eq!(__crc32h(0x00000000, 0x0001), 0x191b3141);
    assert_eq!(__crc32h(0xffffffff, 0x1234), 0xf6b56fbf);
    assert_eq!(__crc32h(0x8ecec3b5, 0x022b), 0x03a1db7c);

    // __crc32w: 32-bit input
    assert_eq!(__crc32w(0x00000000, 0x00000001), 0xb8bc6765);
    assert_eq!(__crc32w(0xffffffff, 0x12345678), 0x5092782d);
    assert_eq!(__crc32w(0xae2912c8, 0x00845fed), 0xc5690dd4);

    // __crc32d: 64-bit input
    assert_eq!(__crc32d(0x00000000, 0x0000000000000001), 0xccaa009e);
    assert_eq!(__crc32d(0xffffffff, 0x123456789abcdef0), 0xe6ddf8b5);
    assert_eq!(__crc32d(0x0badeafe, 0xc0febeefdadafefe), 0x61a45fba);
}

#[target_feature(enable = "crc")]
unsafe fn test_crc32c_castagnoli() {
    // __crc32cb: 8-bit input
    assert_eq!(__crc32cb(0x00000000, 0x01), 0xf26b8303);
    assert_eq!(__crc32cb(0xffffffff, 0x61), 0x3e2fbccf);
    assert_eq!(__crc32cb(0x2aa1e72b, 0x2a), 0xf24122e4);

    // __crc32ch: 16-bit input
    assert_eq!(__crc32ch(0x00000000, 0x0001), 0x13a29877);
    assert_eq!(__crc32ch(0xffffffff, 0x1234), 0xf13f4cea);
    assert_eq!(__crc32ch(0x8ecec3b5, 0x022b), 0x013bb2fb);

    // __crc32cw: 32-bit input
    assert_eq!(__crc32cw(0x00000000, 0x00000001), 0xdd45aab8);
    assert_eq!(__crc32cw(0xffffffff, 0x12345678), 0x4dece20c);
    assert_eq!(__crc32cw(0xae2912c8, 0x00845fed), 0xffae2ed1);

    // __crc32cd: 64-bit input
    assert_eq!(__crc32cd(0x00000000, 0x0000000000000001), 0x493c7d27);
    assert_eq!(__crc32cd(0xffffffff, 0x123456789abcdef0), 0xd95b664b);
    assert_eq!(__crc32cd(0x0badeafe, 0xc0febeefdadafefe), 0x5b44f54f);
}
