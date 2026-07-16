// We're testing aarch64 target specific features
//@only-target: aarch64
//@compile-flags: -C target-feature=+neon
//@run-native

use std::arch::aarch64::*;
use std::arch::is_aarch64_feature_detected;
use std::mem::transmute;

fn main() {
    assert!(is_aarch64_feature_detected!("neon"));

    unsafe {
        test_vpmaxq_u8();
        test_tbl1_basic();
        test_vpadd();
        test_vpaddl();
        test_vqdmulh();
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
fn test_tbl1_basic() {
    unsafe {
        // table = 0..7
        let table: uint8x8_t = transmute::<[u8; 8], _>([0, 1, 2, 3, 4, 5, 6, 7]);

        // indices
        let idx: uint8x8_t = transmute::<[u8; 8], _>([0, 1, 2, 3, 4, 5, 6, 7]);
        let got = vtbl1_u8(table, idx);
        let got_arr: [u8; 8] = transmute(got);
        assert_eq!(got_arr, [0, 1, 2, 3, 4, 5, 6, 7]);

        // Also try different order and out-of-range indices (8, 255).
        let idx2: uint8x8_t = transmute::<[u8; 8], _>([7, 8, 255, 0, 1, 2, 3, 4]);
        let got2 = vtbl1_u8(table, idx2);
        let got2_arr: [u8; 8] = transmute(got2);
        assert_eq!(got2_arr[0], 7);
        assert_eq!(got2_arr[1], 0); // out-of-range
        assert_eq!(got2_arr[2], 0); // out-of-range
        assert_eq!(&got2_arr[3..8], &[0, 1, 2, 3, 4][..]);
    }

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

#[target_feature(enable = "neon")]
unsafe fn test_vpadd() {
    let a = vld1_s8([1, 2, 3, 4, 5, 6, 7, 8].as_ptr());
    let b = vld1_s8([9, 10, -1, 2, i8::MIN, i8::MIN, i8::MAX, i8::MAX].as_ptr());
    let e =
        [3i8, 7, 11, 15, 19, -1 + 2, i8::MIN.wrapping_add(i8::MIN), i8::MAX.wrapping_add(i8::MAX)];
    let mut r = [0i8; 8];
    vst1_s8(r.as_mut_ptr(), vpadd_s8(a, b));
    assert_eq!(r, e);

    let a = vld1_s16([1, 2, 3, 4].as_ptr());
    let b = vld1_s16([-1, 2, i16::MAX, i16::MAX].as_ptr());
    let e = [3i16, 7, -1 + 2, i16::MAX.wrapping_add(i16::MAX)];
    let mut r = [0i16; 4];
    vst1_s16(r.as_mut_ptr(), vpadd_s16(a, b));
    assert_eq!(r, e);

    let a = vld1_s32([1, 2].as_ptr());
    let b = vld1_s32([i32::MAX, i32::MAX].as_ptr());
    let e = [3i32, i32::MAX.wrapping_add(i32::MAX)];
    let mut r = [0i32; 2];
    vst1_s32(r.as_mut_ptr(), vpadd_s32(a, b));
    assert_eq!(r, e);

    let a = vld1_u8([1, 2, 3, 4, 5, 6, 7, 8].as_ptr());
    let b = vld1_u8([9, 10, 11, 12, 13, 14, u8::MAX, u8::MAX].as_ptr());
    let e = [3u8, 7, 11, 15, 19, 23, 27, 254];
    let mut r = [0u8; 8];
    vst1_u8(r.as_mut_ptr(), vpadd_u8(a, b));
    assert_eq!(r, e);

    let a = vld1_u16([1, 2, 3, 4].as_ptr());
    let b = vld1_u16([5, 6, u16::MAX, u16::MAX].as_ptr());
    let e = [3u16, 7, 11, 65534];
    let mut r = [0u16; 4];
    vst1_u16(r.as_mut_ptr(), vpadd_u16(a, b));
    assert_eq!(r, e);

    let a = vld1_u32([1, 2].as_ptr());
    let b = vld1_u32([u32::MAX, u32::MAX].as_ptr());
    let e = [3u32, u32::MAX.wrapping_add(u32::MAX)];
    let mut r = [0u32; 2];
    vst1_u32(r.as_mut_ptr(), vpadd_u32(a, b));
    assert_eq!(r, e);
}

#[target_feature(enable = "neon")]
unsafe fn test_vpaddl() {
    let a = vld1_u8([1, 2, 3, 4, 5, 6, u8::MAX, u8::MAX].as_ptr());
    let e = [3u16, 7, 11, 510];
    let mut r = [0u16; 4];
    vst1_u16(r.as_mut_ptr(), vpaddl_u8(a));
    assert_eq!(r, e);

    let a = vld1q_u8([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, u8::MAX, u8::MAX].as_ptr());
    let e = [3u16, 7, 11, 15, 19, 23, 27, 510];
    let mut r = [0u16; 8];
    vst1q_u16(r.as_mut_ptr(), vpaddlq_u8(a));
    assert_eq!(r, e);

    let a = vld1_u16([1, 2, u16::MAX, u16::MAX].as_ptr());
    let e = [3u32, 131070];
    let mut r = [0u32; 2];
    vst1_u32(r.as_mut_ptr(), vpaddl_u16(a));
    assert_eq!(r, e);

    let a = vld1q_u16([1, 2, 3, 4, 5, 6, u16::MAX, u16::MAX].as_ptr());
    let e = [3u32, 7, 11, 131070];
    let mut r = [0u32; 4];
    vst1q_u32(r.as_mut_ptr(), vpaddlq_u16(a));
    assert_eq!(r, e);

    let a = vld1_u32([1, 2].as_ptr());
    let e = [3u64];
    let mut r = [0u64; 1];
    vst1_u64(r.as_mut_ptr(), vpaddl_u32(a));
    assert_eq!(r, e);

    let a = vld1_u32([u32::MAX, u32::MAX].as_ptr());
    let e = [8589934590];
    let mut r = [0u64; 1];
    vst1_u64(r.as_mut_ptr(), vpaddl_u32(a));
    assert_eq!(r, e);

    let a = vld1q_u32([1, 2, u32::MAX, u32::MAX].as_ptr());
    let e = [3u64, 8589934590];
    let mut r = [0u64; 2];
    vst1q_u64(r.as_mut_ptr(), vpaddlq_u32(a));
    assert_eq!(r, e);
}

#[target_feature(enable = "neon")]
unsafe fn test_vqdmulh() {
    let a = vld1_s32([i32::MIN, i32::MAX].as_ptr());
    let r: [i32; 2] = transmute(vqdmulh_n_s32(a, i32::MIN));
    assert_eq!(r, [i32::MAX, -i32::MAX]);

    // This is the actual calculation that happens.
    let product = i32::MIN as i128 * i32::MIN as i128 * 2;
    assert_eq!(i32::MAX, (product >> 32).clamp(i32::MIN as i128, i32::MAX as i128) as i32);

    let product = i32::MAX as i128 * i32::MIN as i128 * 2;
    assert_eq!(-i32::MAX, (product >> 32).clamp(i32::MIN as i128, i32::MAX as i128) as i32);

    let b = vld1_s32([123, i32::MIN].as_ptr());
    let r: [i32; 2] = transmute(vqdmulh_s32(a, b));
    assert_eq!(r, [-123, -i32::MAX]);

    // Wider 32-bit versions.
    let a = vld1q_s32([0x4000_0000, -0x4000_0000, i32::MIN, i32::MAX].as_ptr());

    let b = vld1q_s32([123, 456, 0x4000_0000, 789].as_ptr());
    let r: [i32; 4] = transmute(vqdmulhq_s32(a, b));
    assert_eq!(r, [61, -228, -1073741824, 788]);

    let r: [i32; 4] = transmute(vqdmulhq_n_s32(a, 0x4000_0000));
    assert_eq!(r, [536870912, -536870912, -1073741824, 1073741823]);

    // 16-bit versions.

    let a = vld1_s16([i16::MIN, i16::MAX, 0, 16384].as_ptr());
    let r: [i16; 4] = transmute(vqdmulh_n_s16(a, i16::MIN));
    assert_eq!(r, [i16::MAX, -i16::MAX, 0, -16384]);

    let b = vld1_s16([123, i16::MIN, 456, 789].as_ptr());
    let r: [i16; 4] = transmute(vqdmulh_s16(a, b));
    assert_eq!(r, [-123, -i16::MAX, 0, 394]);

    // Wider 16-bit versions.

    let a = vld1q_s16([i16::MIN, i16::MAX, 0, 16384, -16384, 8192, 1, -1].as_ptr());
    let b = vld1q_s16([123, 456, 789, i16::MIN, 1, 2, 3, 4].as_ptr());
    let r: [i16; 8] = transmute(vqdmulhq_s16(a, b));
    assert_eq!(r, [-123, 455, 0, -16384, -1, 0, 0, -1]);

    let r: [i16; 8] = transmute(vqdmulhq_n_s16(a, i16::MIN));
    assert_eq!(r, [32767, -32767, 0, -16384, 16384, -8192, -1, 1]);
}
