// We're testing aarch64 target specific features
//@only-target: aarch64
//@compile-flags: -C target-feature=+neon

use std::arch::aarch64::*;
use std::arch::is_aarch64_feature_detected;

fn main() {
    assert!(is_aarch64_feature_detected!("neon"));

    unsafe {
        test_neon();
    }
}

#[target_feature(enable = "neon")]
unsafe fn test_neon() {
    // Adapted from library/stdarch/crates/core_arch/src/aarch64/neon/mod.rs
    unsafe fn test_vpmaxq_u8() {
        let a = vld1q_u8([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8].as_ptr());
        let b = vld1q_u8([0, 3, 2, 5, 4, 7, 6, 9, 0, 3, 2, 5, 4, 7, 6, 9].as_ptr());
        let e = [2, 4, 6, 8, 2, 4, 6, 8, 3, 5, 7, 9, 3, 5, 7, 9];
        let mut r = [0; 16];
        vst1q_u8(r.as_mut_ptr(), vpmaxq_u8(a, b));
        assert_eq!(r, e);
    }
    test_vpmaxq_u8();

    unsafe fn test_vpmaxq_u8_is_unsigned() {
        let a = vld1q_u8(
            [255, 0, 253, 252, 251, 250, 249, 248, 255, 254, 253, 252, 251, 250, 249, 248].as_ptr(),
        );
        let b = vld1q_u8([254, 3, 2, 5, 4, 7, 6, 9, 0, 3, 2, 5, 4, 7, 6, 9].as_ptr());
        let e = [255, 253, 251, 249, 255, 253, 251, 249, 254, 5, 7, 9, 3, 5, 7, 9];
        let mut r = [0; 16];
        vst1q_u8(r.as_mut_ptr(), vpmaxq_u8(a, b));
        assert_eq!(r, e);
    }
    test_vpmaxq_u8_is_unsigned();
}
