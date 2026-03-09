// We're testing aarch64 target specific features
//@only-target: aarch64
//@compile-flags: -C target-feature=+neon

use std::arch::aarch64::*;
use std::arch::is_aarch64_feature_detected;
use std::mem::transmute;

fn main() {
    assert!(is_aarch64_feature_detected!("neon"));

    unsafe {
        test_vpmaxq_u8();
        test_tbl1_v16i8_basic();
    }
}

#[target_feature(enable = "neon")]
unsafe fn test_vpmaxq_u8() {
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

#[target_feature(enable = "neon")]
fn test_tbl1_v16i8_basic() {
    unsafe {
        // table = 0..15
        let table: uint8x16_t =
            transmute::<[u8; 16], _>([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        // indices
        let idx: uint8x16_t =
            transmute::<[u8; 16], _>([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let got = vqtbl1q_u8(table, idx);
        let got_arr: [u8; 16] = transmute(got);
        assert_eq!(got_arr, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

        // Also try different order and out-of-range indices (16, 255).
        let idx2: uint8x16_t =
            transmute::<[u8; 16], _>([15, 16, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let got2 = vqtbl1q_u8(table, idx2);
        let got2_arr: [u8; 16] = transmute(got2);
        assert_eq!(got2_arr[0], 15);
        assert_eq!(got2_arr[1], 0); // out-of-range
        assert_eq!(got2_arr[2], 0); // out-of-range
        assert_eq!(&got2_arr[3..16], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12][..]);
    }
}
