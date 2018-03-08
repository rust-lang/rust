//! ARMv8 ASIMD intrinsics

// FIXME: replace neon with asimd

#[cfg(test)]
use stdsimd_test::assert_instr;
use coresimd::simd_llvm::simd_add;
use coresimd::simd::*;

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fadd))]
pub unsafe fn vadd_f64(a: f64, b: f64) -> f64 {
    a + b
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fadd))]
pub unsafe fn vaddq_f64(a: f64x2, b: f64x2) -> f64x2 {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vaddd_s64(a: i64, b: i64) -> i64 {
    a + b
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vaddd_u64(a: u64, b: u64) -> u64 {
    a + b
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.aarch64.neon.smaxv.i8.v8i8"]
    fn vmaxv_s8_(a: i8x8) -> i8;
    #[link_name = "llvm.aarch64.neon.smaxv.i8.6i8"]
    fn vmaxvq_s8_(a: i8x16) -> i8;
    #[link_name = "llvm.aarch64.neon.smaxv.i16.v4i16"]
    fn vmaxv_s16_(a: i16x4) -> i16;
    #[link_name = "llvm.aarch64.neon.smaxv.i16.v8i16"]
    fn vmaxvq_s16_(a: i16x8) -> i16;
    #[link_name = "llvm.aarch64.neon.smaxv.i32.v2i32"]
    fn vmaxv_s32_(a: i32x2) -> i32;
    #[link_name = "llvm.aarch64.neon.smaxv.i32.v4i32"]
    fn vmaxvq_s32_(a: i32x4) -> i32;

    #[link_name = "llvm.aarch64.neon.umaxv.i8.v8i8"]
    fn vmaxv_u8_(a: u8x8) -> u8;
    #[link_name = "llvm.aarch64.neon.umaxv.i8.6i8"]
    fn vmaxvq_u8_(a: u8x16) -> u8;
    #[link_name = "llvm.aarch64.neon.umaxv.i16.v4i16"]
    fn vmaxv_u16_(a: u16x4) -> u16;
    #[link_name = "llvm.aarch64.neon.umaxv.i16.v8i16"]
    fn vmaxvq_u16_(a: u16x8) -> u16;
    #[link_name = "llvm.aarch64.neon.umaxv.i32.v2i32"]
    fn vmaxv_u32_(a: u32x2) -> u32;
    #[link_name = "llvm.aarch64.neon.umaxv.i32.v4i32"]
    fn vmaxvq_u32_(a: u32x4) -> u32;

    #[link_name = "llvm.aarch64.neon.fmaxv.f32.v2f32"]
    fn vmaxv_f32_(a: f32x2) -> f32;
    #[link_name = "llvm.aarch64.neon.fmaxv.f32.v4f32"]
    fn vmaxvq_f32_(a: f32x4) -> f32;
    #[link_name = "llvm.aarch64.neon.fmaxv.f64.v2f64"]
    fn vmaxvq_f64_(a: f64x2) -> f64;

    #[link_name = "llvm.aarch64.neon.sminv.i8.v8i8"]
    fn vminv_s8_(a: i8x8) -> i8;
    #[link_name = "llvm.aarch64.neon.sminv.i8.6i8"]
    fn vminvq_s8_(a: i8x16) -> i8;
    #[link_name = "llvm.aarch64.neon.sminv.i16.v4i16"]
    fn vminv_s16_(a: i16x4) -> i16;
    #[link_name = "llvm.aarch64.neon.sminv.i16.v8i16"]
    fn vminvq_s16_(a: i16x8) -> i16;
    #[link_name = "llvm.aarch64.neon.sminv.i32.v2i32"]
    fn vminv_s32_(a: i32x2) -> i32;
    #[link_name = "llvm.aarch64.neon.sminv.i32.v4i32"]
    fn vminvq_s32_(a: i32x4) -> i32;

    #[link_name = "llvm.aarch64.neon.uminv.i8.v8i8"]
    fn vminv_u8_(a: u8x8) -> u8;
    #[link_name = "llvm.aarch64.neon.uminv.i8.6i8"]
    fn vminvq_u8_(a: u8x16) -> u8;
    #[link_name = "llvm.aarch64.neon.uminv.i16.v4i16"]
    fn vminv_u16_(a: u16x4) -> u16;
    #[link_name = "llvm.aarch64.neon.uminv.i16.v8i16"]
    fn vminvq_u16_(a: u16x8) -> u16;
    #[link_name = "llvm.aarch64.neon.uminv.i32.v2i32"]
    fn vminv_u32_(a: u32x2) -> u32;
    #[link_name = "llvm.aarch64.neon.uminv.i32.v4i32"]
    fn vminvq_u32_(a: u32x4) -> u32;

    #[link_name = "llvm.aarch64.neon.fminv.f32.v2f32"]
    fn vminv_f32_(a: f32x2) -> f32;
    #[link_name = "llvm.aarch64.neon.fminv.f32.v4f32"]
    fn vminvq_f32_(a: f32x4) -> f32;
    #[link_name = "llvm.aarch64.neon.fminv.f64.v2f64"]
    fn vminvq_f64_(a: f64x2) -> f64;

}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smaxv))]
pub unsafe fn vmaxv_s8(a: i8x8) -> i8 {
    vmaxv_s8_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smaxv))]
pub unsafe fn vmaxvq_s8(a: i8x16) -> i8 {
    vmaxvq_s8_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smaxv))]
pub unsafe fn vmaxv_s16(a: i16x4) -> i16 {
    vmaxv_s16_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smaxv))]
pub unsafe fn vmaxvq_s16(a: i16x8) -> i16 {
    vmaxvq_s16_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smaxp))]
pub unsafe fn vmaxv_s32(a: i32x2) -> i32 {
    vmaxv_s32_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smaxv))]
pub unsafe fn vmaxvq_s32(a: i32x4) -> i32 {
    vmaxvq_s32_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umaxv))]
pub unsafe fn vmaxv_u8(a: u8x8) -> u8 {
    vmaxv_u8_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umaxv))]
pub unsafe fn vmaxvq_u8(a: u8x16) -> u8 {
    vmaxvq_u8_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umaxv))]
pub unsafe fn vmaxv_u16(a: u16x4) -> u16 {
    vmaxv_u16_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umaxv))]
pub unsafe fn vmaxvq_u16(a: u16x8) -> u16 {
    vmaxvq_u16_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umaxp))]
pub unsafe fn vmaxv_u32(a: u32x2) -> u32 {
    vmaxv_u32_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umaxv))]
pub unsafe fn vmaxvq_u32(a: u32x4) -> u32 {
    vmaxvq_u32_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmaxp))]
pub unsafe fn vmaxv_f32(a: f32x2) -> f32 {
    vmaxv_f32_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmaxv))]
pub unsafe fn vmaxvq_f32(a: f32x4) -> f32 {
    vmaxvq_f32_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmaxp))]
pub unsafe fn vmaxvq_f64(a: f64x2) -> f64 {
    vmaxvq_f64_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sminv))]
pub unsafe fn vminv_s8(a: i8x8) -> i8 {
    vminv_s8_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sminv))]
pub unsafe fn vminvq_s8(a: i8x16) -> i8 {
    vminvq_s8_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sminv))]
pub unsafe fn vminv_s16(a: i16x4) -> i16 {
    vminv_s16_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sminv))]
pub unsafe fn vminvq_s16(a: i16x8) -> i16 {
    vminvq_s16_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sminp))]
pub unsafe fn vminv_s32(a: i32x2) -> i32 {
    vminv_s32_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sminv))]
pub unsafe fn vminvq_s32(a: i32x4) -> i32 {
    vminvq_s32_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uminv))]
pub unsafe fn vminv_u8(a: u8x8) -> u8 {
    vminv_u8_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uminv))]
pub unsafe fn vminvq_u8(a: u8x16) -> u8 {
    vminvq_u8_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uminv))]
pub unsafe fn vminv_u16(a: u16x4) -> u16 {
    vminv_u16_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uminv))]
pub unsafe fn vminvq_u16(a: u16x8) -> u16 {
    vminvq_u16_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uminp))]
pub unsafe fn vminv_u32(a: u32x2) -> u32 {
    vminv_u32_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uminv))]
pub unsafe fn vminvq_u32(a: u32x4) -> u32 {
    vminvq_u32_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminp))]
pub unsafe fn vminv_f32(a: f32x2) -> f32 {
    vminv_f32_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminv))]
pub unsafe fn vminvq_f32(a: f32x4) -> f32 {
    vminvq_f32_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminp))]
pub unsafe fn vminvq_f64(a: f64x2) -> f64 {
    vminvq_f64_(a)
}

#[cfg(test)]
mod tests {
    use simd::*;
    use coresimd::aarch64::neon;
    use stdsimd_test::simd_test;

    #[simd_test = "neon"]
    unsafe fn vadd_f64() {
        let a = 1.;
        let b = 8.;
        let e = 9.;
        let r = neon::vadd_f64(a, b);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vaddq_f64() {
        let a = f64x2::new(1., 2.);
        let b = f64x2::new(8., 7.);
        let e = f64x2::new(9., 9.);
        let r = neon::vaddq_f64(a, b);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vaddd_s64() {
        let a = 1;
        let b = 8;
        let e = 9;
        let r = neon::vaddd_s64(a, b);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vaddd_u64() {
        let a = 1;
        let b = 8;
        let e = 9;
        let r = neon::vaddd_u64(a, b);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vmaxv_s8() {
        let r = neon::vmaxv_s8(i8x8::new(1, 2, 3, 4, -8, 6, 7, 5));
        assert_eq!(r, 7_i8);
    }

    #[simd_test = "neon"]
    unsafe fn vmaxvq_s8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let r = neon::vmaxvq_s8(i8x16::new(
            1, 2, 3, 4,
            -16, 6, 7, 5,
            8, 1, 1, 1,
            1, 1, 1, 1,
        ));
        assert_eq!(r, 8_i8);
    }

    #[simd_test = "neon"]
    unsafe fn vmaxv_s16() {
        let r = neon::vmaxv_s16(i16x4::new(1, 2, -4, 3));
        assert_eq!(r, 3_i16);
    }

    #[simd_test = "neon"]
    unsafe fn vmaxvq_s16() {
        let r = neon::vmaxvq_s16(i16x8::new(1, 2, 7, 4, -16, 6, 7, 5));
        assert_eq!(r, 7_i16);
    }

    #[simd_test = "neon"]
    unsafe fn vmaxv_s32() {
        let r = neon::vmaxv_s32(i32x2::new(1, -4));
        assert_eq!(r, 1_i32);
    }

    #[simd_test = "neon"]
    unsafe fn vmaxvq_s32() {
        let r = neon::vmaxvq_s32(i32x4::new(1, 2, -32, 4));
        assert_eq!(r, 4_i32);
    }

    #[simd_test = "neon"]
    unsafe fn vmaxv_u8() {
        let r = neon::vmaxv_u8(u8x8::new(1, 2, 3, 4, 8, 6, 7, 5));
        assert_eq!(r, 8_u8);
    }

    #[simd_test = "neon"]
    unsafe fn vmaxvq_u8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let r = neon::vmaxvq_u8(u8x16::new(
            1, 2, 3, 4,
            16, 6, 7, 5,
            8, 1, 1, 1,
            1, 1, 1, 1,
        ));
        assert_eq!(r, 16_u8);
    }

    #[simd_test = "neon"]
    unsafe fn vmaxv_u16() {
        let r = neon::vmaxv_u16(u16x4::new(1, 2, 4, 3));
        assert_eq!(r, 4_u16);
    }

    #[simd_test = "neon"]
    unsafe fn vmaxvq_u16() {
        let r = neon::vmaxvq_u16(u16x8::new(1, 2, 7, 4, 16, 6, 7, 5));
        assert_eq!(r, 16_u16);
    }

    #[simd_test = "neon"]
    unsafe fn vmaxv_u32() {
        let r = neon::vmaxv_u32(u32x2::new(1, 4));
        assert_eq!(r, 4_u32);
    }

    #[simd_test = "neon"]
    unsafe fn vmaxvq_u32() {
        let r = neon::vmaxvq_u32(u32x4::new(1, 2, 32, 4));
        assert_eq!(r, 32_u32);
    }

    #[simd_test = "neon"]
    unsafe fn vmaxv_f32() {
        let r = neon::vmaxv_f32(f32x2::new(1., 4.));
        assert_eq!(r, 4_f32);
    }

    #[simd_test = "neon"]
    unsafe fn vmaxvq_f32() {
        let r = neon::vmaxvq_f32(f32x4::new(1., 2., 32., 4.));
        assert_eq!(r, 32_f32);
    }

    #[simd_test = "neon"]
    unsafe fn vmaxvq_f64() {
        let r = neon::vmaxvq_f64(f64x2::new(1., 4.));
        assert_eq!(r, 4_f64);
    }

    #[simd_test = "neon"]
    unsafe fn vminv_s8() {
        let r = neon::vminv_s8(i8x8::new(1, 2, 3, 4, -8, 6, 7, 5));
        assert_eq!(r, -8_i8);
    }

    #[simd_test = "neon"]
    unsafe fn vminvq_s8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let r = neon::vminvq_s8(i8x16::new(
            1, 2, 3, 4,
            -16, 6, 7, 5,
            8, 1, 1, 1,
            1, 1, 1, 1,
        ));
        assert_eq!(r, -16_i8);
    }

    #[simd_test = "neon"]
    unsafe fn vminv_s16() {
        let r = neon::vminv_s16(i16x4::new(1, 2, -4, 3));
        assert_eq!(r, -4_i16);
    }

    #[simd_test = "neon"]
    unsafe fn vminvq_s16() {
        let r = neon::vminvq_s16(i16x8::new(1, 2, 7, 4, -16, 6, 7, 5));
        assert_eq!(r, -16_i16);
    }

    #[simd_test = "neon"]
    unsafe fn vminv_s32() {
        let r = neon::vminv_s32(i32x2::new(1, -4));
        assert_eq!(r, -4_i32);
    }

    #[simd_test = "neon"]
    unsafe fn vminvq_s32() {
        let r = neon::vminvq_s32(i32x4::new(1, 2, -32, 4));
        assert_eq!(r, -32_i32);
    }

    #[simd_test = "neon"]
    unsafe fn vminv_u8() {
        let r = neon::vminv_u8(u8x8::new(1, 2, 3, 4, 8, 6, 7, 5));
        assert_eq!(r, 1_u8);
    }

    #[simd_test = "neon"]
    unsafe fn vminvq_u8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let r = neon::vminvq_u8(u8x16::new(
            1, 2, 3, 4,
            16, 6, 7, 5,
            8, 1, 1, 1,
            1, 1, 1, 1,
        ));
        assert_eq!(r, 1_u8);
    }

    #[simd_test = "neon"]
    unsafe fn vminv_u16() {
        let r = neon::vminv_u16(u16x4::new(1, 2, 4, 3));
        assert_eq!(r, 1_u16);
    }

    #[simd_test = "neon"]
    unsafe fn vminvq_u16() {
        let r = neon::vminvq_u16(u16x8::new(1, 2, 7, 4, 16, 6, 7, 5));
        assert_eq!(r, 1_u16);
    }

    #[simd_test = "neon"]
    unsafe fn vminv_u32() {
        let r = neon::vminv_u32(u32x2::new(1, 4));
        assert_eq!(r, 1_u32);
    }

    #[simd_test = "neon"]
    unsafe fn vminvq_u32() {
        let r = neon::vminvq_u32(u32x4::new(1, 2, 32, 4));
        assert_eq!(r, 1_u32);
    }

    #[simd_test = "neon"]
    unsafe fn vminv_f32() {
        let r = neon::vminv_f32(f32x2::new(1., 4.));
        assert_eq!(r, 1_f32);
    }

    #[simd_test = "neon"]
    unsafe fn vminvq_f32() {
        let r = neon::vminvq_f32(f32x4::new(1., 2., 32., 4.));
        assert_eq!(r, 1_f32);
    }

    #[simd_test = "neon"]
    unsafe fn vminvq_f64() {
        let r = neon::vminvq_f64(f64x2::new(1., 4.));
        assert_eq!(r, 1_f64);
    }
}
