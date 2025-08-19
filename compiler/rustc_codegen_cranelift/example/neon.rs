// Most of these tests are copied from https://github.com/japaric/stdsimd/blob/0f4413d01c4f0c3ffbc5a69e9a37fbc7235b31a9/coresimd/arm/neon.rs

#![feature(portable_simd)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
use std::mem::transmute;
#[cfg(target_arch = "aarch64")]
use std::simd::*;

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmin_s8() {
    let a = i8x8::from([1, -2, 3, -4, 5, 6, 7, 8]);
    let b = i8x8::from([0, 3, 2, 5, 4, 7, 6, 9]);
    let e = i8x8::from([-2, -4, 5, 7, 0, 2, 4, 6]);
    let r: i8x8 = transmute(vpmin_s8(transmute(a), transmute(b)));
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmin_s16() {
    let a = i16x4::from([1, 2, 3, -4]);
    let b = i16x4::from([0, 3, 2, 5]);
    let e = i16x4::from([1, -4, 0, 2]);
    let r: i16x4 = transmute(vpmin_s16(transmute(a), transmute(b)));
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmin_s32() {
    let a = i32x2::from([1, -2]);
    let b = i32x2::from([0, 3]);
    let e = i32x2::from([-2, 0]);
    let r: i32x2 = transmute(vpmin_s32(transmute(a), transmute(b)));
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmin_u8() {
    let a = u8x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u8x8::from([0, 3, 2, 5, 4, 7, 6, 9]);
    let e = u8x8::from([1, 3, 5, 7, 0, 2, 4, 6]);
    let r: u8x8 = transmute(vpmin_u8(transmute(a), transmute(b)));
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmin_u16() {
    let a = u16x4::from([1, 2, 3, 4]);
    let b = u16x4::from([0, 3, 2, 5]);
    let e = u16x4::from([1, 3, 0, 2]);
    let r: u16x4 = transmute(vpmin_u16(transmute(a), transmute(b)));
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmin_u32() {
    let a = u32x2::from([1, 2]);
    let b = u32x2::from([0, 3]);
    let e = u32x2::from([1, 0]);
    let r: u32x2 = transmute(vpmin_u32(transmute(a), transmute(b)));
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmin_f32() {
    let a = f32x2::from([1., -2.]);
    let b = f32x2::from([0., 3.]);
    let e = f32x2::from([-2., 0.]);
    let r: f32x2 = transmute(vpmin_f32(transmute(a), transmute(b)));
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmax_s8() {
    let a = i8x8::from([1, -2, 3, -4, 5, 6, 7, 8]);
    let b = i8x8::from([0, 3, 2, 5, 4, 7, 6, 9]);
    let e = i8x8::from([1, 3, 6, 8, 3, 5, 7, 9]);
    let r: i8x8 = transmute(vpmax_s8(transmute(a), transmute(b)));
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmax_s16() {
    let a = i16x4::from([1, 2, 3, -4]);
    let b = i16x4::from([0, 3, 2, 5]);
    let e = i16x4::from([2, 3, 3, 5]);
    let r: i16x4 = transmute(vpmax_s16(transmute(a), transmute(b)));
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmax_s32() {
    let a = i32x2::from([1, -2]);
    let b = i32x2::from([0, 3]);
    let e = i32x2::from([1, 3]);
    let r: i32x2 = transmute(vpmax_s32(transmute(a), transmute(b)));
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmax_u8() {
    let a = u8x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u8x8::from([0, 3, 2, 5, 4, 7, 6, 9]);
    let e = u8x8::from([2, 4, 6, 8, 3, 5, 7, 9]);
    let r: u8x8 = transmute(vpmax_u8(transmute(a), transmute(b)));
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmax_u16() {
    let a = u16x4::from([1, 2, 3, 4]);
    let b = u16x4::from([0, 3, 2, 5]);
    let e = u16x4::from([2, 4, 3, 5]);
    let r: u16x4 = transmute(vpmax_u16(transmute(a), transmute(b)));
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmax_u32() {
    let a = u32x2::from([1, 2]);
    let b = u32x2::from([0, 3]);
    let e = u32x2::from([2, 3]);
    let r: u32x2 = transmute(vpmax_u32(transmute(a), transmute(b)));
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpmax_f32() {
    let a = f32x2::from([1., -2.]);
    let b = f32x2::from([0., 3.]);
    let e = f32x2::from([1., 3.]);
    let r: f32x2 = transmute(vpmax_f32(transmute(a), transmute(b)));
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vpadd_s16() {
    let a = i16x4::from([1, 2, 3, 4]);
    let b = i16x4::from([0, -1, -2, -3]);
    let r: i16x4 = transmute(vpadd_s16(transmute(a), transmute(b)));
    let e = i16x4::from([3, 7, -1, -5]);
    assert_eq!(r, e);
}
#[cfg(target_arch = "aarch64")]
unsafe fn test_vpadd_s32() {
    let a = i32x2::from([1, 2]);
    let b = i32x2::from([0, -1]);
    let r: i32x2 = transmute(vpadd_s32(transmute(a), transmute(b)));
    let e = i32x2::from([3, -1]);
    assert_eq!(r, e);
}
#[cfg(target_arch = "aarch64")]
unsafe fn test_vpadd_s8() {
    let a = i8x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i8x8::from([0, -1, -2, -3, -4, -5, -6, -7]);
    let r: i8x8 = transmute(vpadd_s8(transmute(a), transmute(b)));
    let e = i8x8::from([3, 7, 11, 15, -1, -5, -9, -13]);
    assert_eq!(r, e);
}
#[cfg(target_arch = "aarch64")]
unsafe fn test_vpadd_u16() {
    let a = u16x4::from([1, 2, 3, 4]);
    let b = u16x4::from([30, 31, 32, 33]);
    let r: u16x4 = transmute(vpadd_u16(transmute(a), transmute(b)));
    let e = u16x4::from([3, 7, 61, 65]);
    assert_eq!(r, e);
}
#[cfg(target_arch = "aarch64")]
unsafe fn test_vpadd_u32() {
    let a = u32x2::from([1, 2]);
    let b = u32x2::from([30, 31]);
    let r: u32x2 = transmute(vpadd_u32(transmute(a), transmute(b)));
    let e = u32x2::from([3, 61]);
    assert_eq!(r, e);
}
#[cfg(target_arch = "aarch64")]
unsafe fn test_vpadd_u8() {
    let a = u8x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u8x8::from([30, 31, 32, 33, 34, 35, 36, 37]);
    let r: u8x8 = transmute(vpadd_u8(transmute(a), transmute(b)));
    let e = u8x8::from([3, 7, 11, 15, 61, 65, 69, 73]);
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vqsub_u8() {
    let a = u8x8::from([1, 2, 3, 4, 5, 6, 7, 0xff]);
    let b = u8x8::from([30, 1, 1, 1, 34, 0xff, 36, 37]);
    let r: u8x8 = transmute(vqsub_u8(transmute(a), transmute(b)));
    let e = u8x8::from([0, 1, 2, 3, 0, 0, 0, 218]);
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vqadd_u8() {
    let a = u8x8::from([1, 2, 3, 4, 5, 6, 7, 0xff]);
    let b = u8x8::from([30, 1, 1, 1, 34, 0xff, 36, 37]);
    let r: u8x8 = transmute(vqadd_u8(transmute(a), transmute(b)));
    let e = u8x8::from([31, 3, 4, 5, 39, 0xff, 43, 0xff]);
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vmaxq_f32() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.fmax.v4f32
    let a = f32x4::from([0., -1., 2., -3.]);
    let b = f32x4::from([-4., 5., -6., 7.]);
    let e = f32x4::from([0., 5., 2., 7.]);
    let r: f32x4 = transmute(vmaxq_f32(transmute(a), transmute(b)));
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vminq_f32() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.fmin.v4f32
    let a = f32x4::from([0., -1., 2., -3.]);
    let b = f32x4::from([-4., 5., -6., 7.]);
    let e = f32x4::from([-4., -1., -6., -3.]);
    let r: f32x4 = transmute(vminq_f32(transmute(a), transmute(b)));
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vaddvq_f32() {
    // AArch64 llvm intrinsic: llvm.aarch64.neon.faddv.f32.v4f32
    let a = f32x4::from([0., 1., 2., 3.]);
    let e = 6f32;
    let r = vaddvq_f32(transmute(a));
    assert_eq!(r, e);
}

#[cfg(target_arch = "aarch64")]
unsafe fn test_vrndnq_f32() {
    // llvm intrinsic: llvm.roundeven.v4f32
    let a = f32x4::from([0.1, -1.9, 4.5, 5.5]);
    let e = f32x4::from([0., -2., 4., 6.]);
    let r: f32x4 = transmute(vrndnq_f32(transmute(a)));
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
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {}
