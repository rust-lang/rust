// Most of these tests are copied from https://github.com/japaric/stdsimd/blob/0f4413d01c4f0c3ffbc5a69e9a37fbc7235b31a9/coresimd/arm/neon.rs

#![cfg_attr(target_arch = "aarch64", feature(portable_simd))]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
use std::mem::transmute;
#[cfg(target_arch = "aarch64")]
use std::simd::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn test_crc32() {
    assert!(std::arch::is_aarch64_feature_detected!("crc"));

    let a: u32 = 42;
    let b: u64 = 0xdeadbeef;

    assert_eq!(__crc32b(a, b as u8), 0xEB0E363F);
    assert_eq!(__crc32h(a, b as u16), 0x9A54BD80);
    assert_eq!(__crc32w(a, b as u32), 0xF491F059);
    assert_eq!(__crc32d(a, b as u64), 0xD14BBEA6);

    assert_eq!(__crc32cb(a, b as u8), 0xF67C32D8);
    assert_eq!(__crc32ch(a, b as u16), 0x479108B8);
    assert_eq!(__crc32cw(a, b as u32), 0x979F49F8);
    assert_eq!(__crc32cd(a, b as u64), 0x0E6BE593);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmin_s8() {
    let a = i8x8::from([1, -2, 3, -4, 5, 6, 7, 8]);
    let b = i8x8::from([0, 3, 2, 5, 4, 7, 6, 9]);
    let e = i8x8::from([-2, -4, 5, 7, 0, 2, 4, 6]);
    let r: i8x8 = unsafe { transmute(vpmin_s8(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmin_s16() {
    let a = i16x4::from([1, 2, 3, -4]);
    let b = i16x4::from([0, 3, 2, 5]);
    let e = i16x4::from([1, -4, 0, 2]);
    let r: i16x4 = unsafe { transmute(vpmin_s16(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmin_s32() {
    let a = i32x2::from([1, -2]);
    let b = i32x2::from([0, 3]);
    let e = i32x2::from([-2, 0]);
    let r: i32x2 = unsafe { transmute(vpmin_s32(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmin_u8() {
    let a = u8x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u8x8::from([0, 3, 2, 5, 4, 7, 6, 9]);
    let e = u8x8::from([1, 3, 5, 7, 0, 2, 4, 6]);
    let r: u8x8 = unsafe { transmute(vpmin_u8(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmin_u16() {
    let a = u16x4::from([1, 2, 3, 4]);
    let b = u16x4::from([0, 3, 2, 5]);
    let e = u16x4::from([1, 3, 0, 2]);
    let r: u16x4 = unsafe { transmute(vpmin_u16(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmin_u32() {
    let a = u32x2::from([1, 2]);
    let b = u32x2::from([0, 3]);
    let e = u32x2::from([1, 0]);
    let r: u32x2 = unsafe { transmute(vpmin_u32(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmin_f32() {
    let a = f32x2::from([1., -2.]);
    let b = f32x2::from([0., 3.]);
    let e = f32x2::from([-2., 0.]);
    let r: f32x2 = unsafe { transmute(vpmin_f32(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmax_s8() {
    let a = i8x8::from([1, -2, 3, -4, 5, 6, 7, 8]);
    let b = i8x8::from([0, 3, 2, 5, 4, 7, 6, 9]);
    let e = i8x8::from([1, 3, 6, 8, 3, 5, 7, 9]);
    let r: i8x8 = unsafe { transmute(vpmax_s8(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmax_s16() {
    let a = i16x4::from([1, 2, 3, -4]);
    let b = i16x4::from([0, 3, 2, 5]);
    let e = i16x4::from([2, 3, 3, 5]);
    let r: i16x4 = unsafe { transmute(vpmax_s16(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmax_s32() {
    let a = i32x2::from([1, -2]);
    let b = i32x2::from([0, 3]);
    let e = i32x2::from([1, 3]);
    let r: i32x2 = unsafe { transmute(vpmax_s32(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmax_u8() {
    let a = u8x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u8x8::from([0, 3, 2, 5, 4, 7, 6, 9]);
    let e = u8x8::from([2, 4, 6, 8, 3, 5, 7, 9]);
    let r: u8x8 = unsafe { transmute(vpmax_u8(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmax_u16() {
    let a = u16x4::from([1, 2, 3, 4]);
    let b = u16x4::from([0, 3, 2, 5]);
    let e = u16x4::from([2, 4, 3, 5]);
    let r: u16x4 = unsafe { transmute(vpmax_u16(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmax_u32() {
    let a = u32x2::from([1, 2]);
    let b = u32x2::from([0, 3]);
    let e = u32x2::from([2, 3]);
    let r: u32x2 = unsafe { transmute(vpmax_u32(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmax_f32() {
    let a = f32x2::from([1., -2.]);
    let b = f32x2::from([0., 3.]);
    let e = f32x2::from([1., 3.]);
    let r: f32x2 = unsafe { transmute(vpmax_f32(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpadd_s16() {
    let a = i16x4::from([1, 2, 3, 4]);
    let b = i16x4::from([0, -1, -2, -3]);
    let r: i16x4 = unsafe { transmute(vpadd_s16(transmute(a), transmute(b))) };
    let e = i16x4::from([3, 7, -1, -5]);
    assert_eq!(r, e);
}
#[cfg(target_arch = "aarch64")]
unsafe fn test_vpadd_s32() {
    let a = i32x2::from([1, 2]);
    let b = i32x2::from([0, -1]);
    let r: i32x2 = unsafe { transmute(vpadd_s32(transmute(a), transmute(b))) };
    let e = i32x2::from([3, -1]);
    assert_eq!(r, e);
}
#[cfg(target_arch = "aarch64")]
unsafe fn test_vpadd_s8() {
    let a = i8x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i8x8::from([0, -1, -2, -3, -4, -5, -6, -7]);
    let r: i8x8 = unsafe { transmute(vpadd_s8(transmute(a), transmute(b))) };
    let e = i8x8::from([3, 7, 11, 15, -1, -5, -9, -13]);
    assert_eq!(r, e);
}
#[cfg(target_arch = "aarch64")]
unsafe fn test_vpadd_u16() {
    let a = u16x4::from([1, 2, 3, 4]);
    let b = u16x4::from([30, 31, 32, 33]);
    let r: u16x4 = unsafe { transmute(vpadd_u16(transmute(a), transmute(b))) };
    let e = u16x4::from([3, 7, 61, 65]);
    assert_eq!(r, e);
}
#[cfg(target_arch = "aarch64")]
unsafe fn test_vpadd_u32() {
    let a = u32x2::from([1, 2]);
    let b = u32x2::from([30, 31]);
    let r: u32x2 = unsafe { transmute(vpadd_u32(transmute(a), transmute(b))) };
    let e = u32x2::from([3, 61]);
    assert_eq!(r, e);
}
#[cfg(target_arch = "aarch64")]
unsafe fn test_vpadd_u8() {
    let a = u8x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u8x8::from([30, 31, 32, 33, 34, 35, 36, 37]);
    let r: u8x8 = unsafe { transmute(vpadd_u8(transmute(a), transmute(b))) };
    let e = u8x8::from([3, 7, 11, 15, 61, 65, 69, 73]);
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vqsub_u8() {
    let a = u8x8::from([1, 2, 3, 4, 5, 6, 7, 0xff]);
    let b = u8x8::from([30, 1, 1, 1, 34, 0xff, 36, 37]);
    let r: u8x8 = unsafe { transmute(vqsub_u8(transmute(a), transmute(b))) };
    let e = u8x8::from([0, 1, 2, 3, 0, 0, 0, 218]);
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vqadd_u8() {
    let a = u8x8::from([1, 2, 3, 4, 5, 6, 7, 0xff]);
    let b = u8x8::from([30, 1, 1, 1, 34, 0xff, 36, 37]);
    let r: u8x8 = unsafe { transmute(vqadd_u8(transmute(a), transmute(b))) };
    let e = u8x8::from([31, 3, 4, 5, 39, 0xff, 43, 0xff]);
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vmaxq_f32() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.fmax.v4f32
    let a = f32x4::from([0., -1., 2., -3.]);
    let b = f32x4::from([-4., 5., -6., 7.]);
    let e = f32x4::from([0., 5., 2., 7.]);
    let r: f32x4 = unsafe { transmute(vmaxq_f32(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vminq_f32() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.fmin.v4f32
    let a = f32x4::from([0., -1., 2., -3.]);
    let b = f32x4::from([-4., 5., -6., 7.]);
    let e = f32x4::from([-4., -1., -6., -3.]);
    let r: f32x4 = unsafe { transmute(vminq_f32(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vaddvq_f32() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.faddv.f32.v4f32
    let a = f32x4::from([0., 1., 2., 3.]);
    let e = 6f32;
    let r = unsafe { vaddvq_f32(transmute(a)) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vrndnq_f32() {
    // llvm intrinsic: llvm.roundeven.v4f32
    let a = f32x4::from([0.1, -1.9, 4.5, 5.5]);
    let e = f32x4::from([0., -2., 4., 6.]);
    let r: f32x4 = unsafe { transmute(vrndnq_f32(transmute(a))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn test_vaeseq_u8() {
    // AArch64 llvm intrinsic: llvm.aarch64.crypto.aese
    let a = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    let b = u8x16::from([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]);
    let e = u8x16::from([
        0xca, 0xca, 0xca, 0xca, 0xca, 0xca, 0xca, 0xca, 0xca, 0xca, 0xca, 0xca, 0xca, 0xca, 0xca,
        0xca,
    ]);
    let r: u8x16 = unsafe { transmute(vaeseq_u8(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn test_vaesdq_u8() {
    // AArch64 llvm intrinsic: llvm.aarch64.crypto.aesd
    let a = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    let b = u8x16::from([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]);
    let e = u8x16::from([
        0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c,
        0x7c,
    ]);
    let r: u8x16 = unsafe { transmute(vaesdq_u8(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn test_vaesmcq_u8() {
    // AArch64 llvm intrinsic: llvm.aarch64.crypto.aesmc
    let a = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    let e = u8x16::from([2, 7, 0, 5, 6, 3, 4, 1, 10, 15, 8, 13, 14, 11, 12, 9]);
    let r: u8x16 = unsafe { transmute(vaesmcq_u8(transmute(a))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn test_vaesimcq_u8() {
    // AArch64 llvm intrinsic: llvm.aarch64.crypto.aesimc
    let a = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    let e = u8x16::from([10, 15, 8, 13, 14, 11, 12, 9, 2, 7, 0, 5, 6, 3, 4, 1]);
    let r: u8x16 = unsafe { transmute(vaesimcq_u8(transmute(a))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sha2")]
unsafe fn test_vsha256hq_u32() {
    // AArch64 llvm intrinsic: llvm.aarch64.crypto.sha256h
    let a = u32x4::from([0, 1, 2, 3]);
    let b = u32x4::from([4, 5, 6, 7]);
    let c = u32x4::from([8, 9, 10, 11]);
    let e = u32x4::from([0x27bb4ae0, 0xd8f61f7c, 0xb7c1ecdc, 0x10800215]);
    let r: u32x4 = unsafe { transmute(vsha256hq_u32(transmute(a), transmute(b), transmute(c))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sha2")]
unsafe fn test_vsha256h2q_u32() {
    // AArch64 llvm intrinsic: llvm.aarch64.crypto.sha256h2
    let a = u32x4::from([0, 1, 2, 3]);
    let b = u32x4::from([4, 5, 6, 7]);
    let c = u32x4::from([8, 9, 10, 11]);
    let e = u32x4::from([0x6989ee0d, 0x4b055920, 0x52800a12, 0x00000014]);
    let r: u32x4 = unsafe { transmute(vsha256h2q_u32(transmute(a), transmute(b), transmute(c))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sha2")]
unsafe fn test_vsha256su0q_u32() {
    // AArch64 llvm intrinsic: llvm.aarch64.crypto.sha256su0
    let a = u32x4::from([0, 1, 2, 3]);
    let b = u32x4::from([4, 5, 6, 7]);
    let e = u32x4::from([0x02004000, 0x04008001, 0x0600c002, 0x08010003]);
    let r: u32x4 = unsafe { transmute(vsha256su0q_u32(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sha2")]
unsafe fn test_vsha256su1q_u32() {
    // AArch64 llvm intrinsic: llvm.aarch64.crypto.sha256su1
    let a = u32x4::from([0, 1, 2, 3]);
    let b = u32x4::from([4, 5, 6, 7]);
    let c = u32x4::from([8, 9, 10, 11]);
    let e = u32x4::from([0x00044005, 0x0004e007, 0xa802211b, 0xec036145]);
    let r: u32x4 = unsafe { transmute(vsha256su1q_u32(transmute(a), transmute(b), transmute(c))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
fn test_vmull_p64() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.pmull64
    let a: u64 = 3;
    let b: u64 = 6;
    let e: u128 = 10;
    let r: u128 = vmull_p64(a, b);
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vmull_p8() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.pmull.v8i16
    let a = u8x8::from([0, 1, 2, 3, 4, 5, 6, 7]);
    let b = u8x8::from([8, 9, 10, 11, 12, 13, 14, 15]);
    let e = u16x8::from([0x0000, 0x0009, 0x0014, 0x001d, 0x0030, 0x0039, 0x0024, 0x002d]);
    let r: u16x8 = unsafe { transmute(vmull_p8(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vqdmulh_s16() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.sqdmulh.v4i16
    let a = i16x4::from([1, 2, 4, 8]);
    let b = i16x4::from([16384, 16384, 16384, 16384]);
    let e = i16x4::from([0, 1, 2, 4]);
    let r: i16x4 = unsafe { transmute(vqdmulh_s16(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vqdmulh_s32() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.sqdmulh.v2i32
    let a = i32x2::from([1, 2]);
    let b = i32x2::from([1073741824, 1073741824]);
    let e = i32x2::from([0, 1]);
    let r: i32x2 = unsafe { transmute(vqdmulh_s32(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vqdmulhq_s16() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.sqdmulh.v8i16
    let a = i16x8::from([1, 2, 4, 8, 16, 32, 64, 128]);
    let b = i16x8::from([16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384]);
    let e = i16x8::from([0, 1, 2, 4, 8, 16, 32, 64]);
    let r: i16x8 = unsafe { transmute(vqdmulhq_s16(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vqdmulhq_s32() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.sqdmulh.v4i32
    let a = i32x4::from([1, 2, 4, 8]);
    let b = i32x4::from([1073741824, 1073741824, 1073741824, 1073741824]);
    let e = i32x4::from([0, 1, 2, 4]);
    let r: i32x4 = unsafe { transmute(vqdmulhq_s32(transmute(a), transmute(b))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpaddl_s8() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.saddlp.v4i16.v8i8
    let a = i8x8::from([1, 2, 3, 4, -5, -6, -7, -8]);
    let e = i16x4::from([3, 7, -11, -15]);
    let r: i16x4 = unsafe { transmute(vpaddl_s8(transmute(a))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpaddl_s16() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.saddlp.v2i32.v4i16
    let a = i16x4::from([1, 2, -3, -4]);
    let e = i32x2::from([3, -7]);
    let r: i32x2 = unsafe { transmute(vpaddl_s16(transmute(a))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpaddl_s32() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.saddlp.v1i64.v2i32
    let a = i32x2::from([1, -2]);
    let e = i64x1::from([-1]);
    let r: i64x1 = unsafe { transmute(vpaddl_s32(transmute(a))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpaddlq_s8() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.saddlp.v8i16.v16i8
    let a = i8x16::from([1, 2, 3, 4, 5, 6, 7, 8, -9, -10, -11, -12, -13, -14, -15, -16]);
    let e = i16x8::from([3, 7, 11, 15, -19, -23, -27, -31]);
    let r: i16x8 = unsafe { transmute(vpaddlq_s8(transmute(a))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpaddlq_s16() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.saddlp.v4i32.v8i16
    let a = i16x8::from([1, 2, 3, 4, -5, -6, -7, -8]);
    let e = i32x4::from([3, 7, -11, -15]);
    let r: i32x4 = unsafe { transmute(vpaddlq_s16(transmute(a))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpaddlq_s32() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.saddlp.v2i64.v4i32
    let a = i32x4::from([1, 2, -3, -4]);
    let e = i64x2::from([3, -7]);
    let r: i64x2 = unsafe { transmute(vpaddlq_s32(transmute(a))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpaddl_u8() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.uaddlp.v4i16.v8i8
    let a = u8x8::from([255, 254, 253, 252, 251, 250, 249, 248]);
    let e = u16x4::from([509, 505, 501, 497]);
    let r: u16x4 = unsafe { transmute(vpaddl_u8(transmute(a))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpaddl_u16() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.uaddlp.v2i32.v4i16
    let a = u16x4::from([65535, 65534, 65533, 65532]);
    let e = u32x2::from([131069, 131065]);
    let r: u32x2 = unsafe { transmute(vpaddl_u16(transmute(a))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpaddl_u32() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.uaddlp.v1i64.v2i32
    let a = u32x2::from([4294967295, 4294967294]);
    let e = u64x1::from([8589934589]);
    let r: u64x1 = unsafe { transmute(vpaddl_u32(transmute(a))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpaddlq_u8() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.uaddlp.v8i16.v16i8
    let a = u8x16::from([
        255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240,
    ]);
    let e = u16x8::from([509, 505, 501, 497, 493, 489, 485, 481]);
    let r: u16x8 = unsafe { transmute(vpaddlq_u8(transmute(a))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpaddlq_u16() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.uaddlp.v4i32.v8i16
    let a = u16x8::from([65535, 65534, 65533, 65532, 65531, 65530, 65529, 65528]);
    let e = u32x4::from([131069, 131065, 131061, 131057]);
    let r: u32x4 = unsafe { transmute(vpaddlq_u16(transmute(a))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpaddlq_u32() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.uaddlp.v2i64.v4i32
    let a = u32x4::from([4294967295, 4294967294, 4294967293, 4294967292]);
    let e = u64x2::from([8589934589, 8589934585]);
    let r: u64x2 = unsafe { transmute(vpaddlq_u32(transmute(a))) };
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
fn main() {
    unsafe {
        test_vpmin_s8();
        test_vpmin_s16();
        test_vpmin_s32();
        test_vpmin_u8();
        test_vpmin_u16();
        test_vpmin_u32();
        test_vpmin_f32();
        test_vpmax_s8();
        test_vpmax_s16();
        test_vpmax_s32();
        test_vpmax_u8();
        test_vpmax_u16();
        test_vpmax_u32();
        test_vpmax_f32();

        test_vpadd_s16();
        test_vpadd_s32();
        test_vpadd_s8();
        test_vpadd_u16();
        test_vpadd_u32();
        test_vpadd_u8();

        test_vqsub_u8();
        test_vqadd_u8();

        test_vmaxq_f32();
        test_vminq_f32();
        test_vaddvq_f32();
        test_vrndnq_f32();

        test_crc32();

        test_vaeseq_u8();
        test_vaesdq_u8();
        test_vaesmcq_u8();
        test_vaesimcq_u8();

        test_vsha256hq_u32();
        test_vsha256h2q_u32();
        test_vsha256su0q_u32();
        test_vsha256su1q_u32();

        test_vmull_p64();
        test_vmull_p8();

        test_vqdmulh_s16();
        test_vqdmulh_s32();
        test_vqdmulhq_s16();
        test_vqdmulhq_s32();

        test_vpaddl_s8();
        test_vpaddl_s16();
        test_vpaddl_s32();
        test_vpaddlq_s8();
        test_vpaddlq_s16();
        test_vpaddlq_s32();

        test_vpaddl_u8();
        test_vpaddl_u16();
        test_vpaddl_u32();
        test_vpaddlq_u8();
        test_vpaddlq_u16();
        test_vpaddlq_u32();
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {}
