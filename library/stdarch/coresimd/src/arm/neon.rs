//! ARMv7 NEON intrinsics

#[cfg(test)]
use stdsimd_test::assert_instr;

use simd_llvm::simd_add;

use v64::{f32x2, i16x4, i32x2, i8x8, u16x4, u32x2, u8x8};
use v128::{f32x4, i16x8, i32x4, i64x2, i8x16, u16x8, u32x4, u64x2, u8x16};

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vadd_s8(a: i8x8, b: i8x8) -> i8x8 {
    simd_add(a, b)
}

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vaddq_s8(a: i8x16, b: i8x16) -> i8x16 {
    simd_add(a, b)
}

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vadd_s16(a: i16x4, b: i16x4) -> i16x4 {
    simd_add(a, b)
}

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vaddq_s16(a: i16x8, b: i16x8) -> i16x8 {
    simd_add(a, b)
}

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vadd_s32(a: i32x2, b: i32x2) -> i32x2 {
    simd_add(a, b)
}

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vaddq_s32(a: i32x4, b: i32x4) -> i32x4 {
    simd_add(a, b)
}

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vaddq_s64(a: i64x2, b: i64x2) -> i64x2 {
    simd_add(a, b)
}

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vadd_u8(a: u8x8, b: u8x8) -> u8x8 {
    simd_add(a, b)
}

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vaddq_u8(a: u8x16, b: u8x16) -> u8x16 {
    simd_add(a, b)
}

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vadd_u16(a: u16x4, b: u16x4) -> u16x4 {
    simd_add(a, b)
}

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vaddq_u16(a: u16x8, b: u16x8) -> u16x8 {
    simd_add(a, b)
}

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vadd_u32(a: u32x2, b: u32x2) -> u32x2 {
    simd_add(a, b)
}

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vaddq_u32(a: u32x4, b: u32x4) -> u32x4 {
    simd_add(a, b)
}

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vaddq_u64(a: u64x2, b: u64x2) -> u64x2 {
    simd_add(a, b)
}

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(fadd))]
pub unsafe fn vadd_f32(a: f32x2, b: f32x2) -> f32x2 {
    simd_add(a, b)
}

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(fadd))]
pub unsafe fn vaddq_f32(a: f32x4, b: f32x4) -> f32x4 {
    simd_add(a, b)
}

/// Vector long add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(saddl))]
pub unsafe fn vaddl_s8(a: i8x8, b: i8x8) -> i16x8 {
    let a = a.as_i16x8();
    let b = b.as_i16x8();
    simd_add(a, b)
}

/// Vector long add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(saddl))]
pub unsafe fn vaddl_s16(a: i16x4, b: i16x4) -> i32x4 {
    let a = a.as_i32x4();
    let b = b.as_i32x4();
    simd_add(a, b)
}

/// Vector long add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(saddl))]
pub unsafe fn vaddl_s32(a: i32x2, b: i32x2) -> i64x2 {
    let a = a.as_i64x2();
    let b = b.as_i64x2();
    simd_add(a, b)
}

/// Vector long add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(uaddl))]
pub unsafe fn vaddl_u8(a: u8x8, b: u8x8) -> u16x8 {
    let a = a.as_u16x8();
    let b = b.as_u16x8();
    simd_add(a, b)
}

/// Vector long add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(uaddl))]
pub unsafe fn vaddl_u16(a: u16x4, b: u16x4) -> u32x4 {
    let a = a.as_u32x4();
    let b = b.as_u32x4();
    simd_add(a, b)
}

/// Vector long add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(uaddl))]
pub unsafe fn vaddl_u32(a: u32x2, b: u32x2) -> u64x2 {
    let a = a.as_u64x2();
    let b = b.as_u64x2();
    simd_add(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    // The Reference says this instruction is
    // supported in v7/A32/A64:
    #[link_name = "llvm.aarch64.neon.frsqrte.v2f32"]
    fn frsqrte_v2f32(a: f32x2) -> f32x2;
}

/// Reciprocal square-root estimate.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(frsqrte))]
pub unsafe fn vrsqrte_f32(a: f32x2) -> f32x2 {
    frsqrte_v2f32(a)
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;
    use simd::*;
    use arm::neon;

    #[simd_test = "neon"]
    unsafe fn vadd_s8() {
        let a = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i8x8::new(8, 7, 6, 5, 4, 3, 2, 1);
        let e = i8x8::new(9, 9, 9, 9, 9, 9, 9, 9);
        let r = neon::vadd_s8(a, b);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vaddq_s8() {
        let a = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8);
        let b = i8x16::new(8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1);
        let e = i8x16::new(9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9);
        let r = neon::vaddq_s8(a, b);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vadd_s16() {
        let a = i16x4::new(1, 2, 3, 4);
        let b = i16x4::new(8, 7, 6, 5);
        let e = i16x4::new(9, 9, 9, 9);
        let r = neon::vadd_s16(a, b);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vaddq_s16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i16x8::new(8, 7, 6, 5, 4, 3, 2, 1);
        let e = i16x8::new(9, 9, 9, 9, 9, 9, 9, 9);
        let r = neon::vaddq_s16(a, b);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vadd_s32() {
        let a = i32x2::new(1, 2);
        let b = i32x2::new(8, 7);
        let e = i32x2::new(9, 9);
        let r = neon::vadd_s32(a, b);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vaddq_s32() {
        let a = i32x4::new(1, 2, 3, 4);
        let b = i32x4::new(8, 7, 6, 5);
        let e = i32x4::new(9, 9, 9, 9);
        let r = neon::vaddq_s32(a, b);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vadd_u8() {
        let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u8x8::new(8, 7, 6, 5, 4, 3, 2, 1);
        let e = u8x8::new(9, 9, 9, 9, 9, 9, 9, 9);
        let r = neon::vadd_u8(a, b);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vaddq_u8() {
        let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8);
        let b = u8x16::new(8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1);
        let e = u8x16::new(9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9);
        let r = neon::vaddq_u8(a, b);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vadd_u16() {
        let a = u16x4::new(1, 2, 3, 4);
        let b = u16x4::new(8, 7, 6, 5);
        let e = u16x4::new(9, 9, 9, 9);
        let r = neon::vadd_u16(a, b);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vaddq_u16() {
        let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u16x8::new(8, 7, 6, 5, 4, 3, 2, 1);
        let e = u16x8::new(9, 9, 9, 9, 9, 9, 9, 9);
        let r = neon::vaddq_u16(a, b);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vadd_u32() {
        let a = u32x2::new(1, 2);
        let b = u32x2::new(8, 7);
        let e = u32x2::new(9, 9);
        let r = neon::vadd_u32(a, b);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vaddq_u32() {
        let a = u32x4::new(1, 2, 3, 4);
        let b = u32x4::new(8, 7, 6, 5);
        let e = u32x4::new(9, 9, 9, 9);
        let r = neon::vaddq_u32(a, b);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vadd_f32() {
        let a = f32x2::new(1., 2.);
        let b = f32x2::new(8., 7.);
        let e = f32x2::new(9., 9.);
        let r = neon::vadd_f32(a, b);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vaddq_f32() {
        let a = f32x4::new(1., 2., 3., 4.);
        let b = f32x4::new(8., 7., 6., 5.);
        let e = f32x4::new(9., 9., 9., 9.);
        let r = neon::vaddq_f32(a, b);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vaddl_s8() {
        let v = ::std::i8::MAX;
        let a = i8x8::new(v, v, v, v, v, v, v, v);
        let v = 2 * (v as i16);
        let e = i16x8::new(v, v, v, v, v, v, v, v);
        let r = neon::vaddl_s8(a, a);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vaddl_s16() {
        let v = ::std::i16::MAX;
        let a = i16x4::new(v, v, v, v);
        let v = 2 * (v as i32);
        let e = i32x4::new(v, v, v, v);
        let r = neon::vaddl_s16(a, a);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vaddl_s32() {
        let v = ::std::i32::MAX;
        let a = i32x2::new(v, v);
        let v = 2 * (v as i64);
        let e = i64x2::new(v, v);
        let r = neon::vaddl_s32(a, a);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vaddl_u8() {
        let v = ::std::u8::MAX;
        let a = u8x8::new(v, v, v, v, v, v, v, v);
        let v = 2 * (v as u16);
        let e = u16x8::new(v, v, v, v, v, v, v, v);
        let r = neon::vaddl_u8(a, a);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vaddl_u16() {
        let v = ::std::u16::MAX;
        let a = u16x4::new(v, v, v, v);
        let v = 2 * (v as u32);
        let e = u32x4::new(v, v, v, v);
        let r = neon::vaddl_u16(a, a);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vaddl_u32() {
        let v = ::std::u32::MAX;
        let a = u32x2::new(v, v);
        let v = 2 * (v as u64);
        let e = u64x2::new(v, v);
        let r = neon::vaddl_u32(a, a);
        assert_eq!(r, e);
    }

    #[simd_test = "neon"]
    unsafe fn vrsqrt_f32() {
        let a = f32x2::new(1.0, 2.0);
        let e = f32x2::new(0.9980469, 0.7050781);
        let r = neon::vrsqrte_f32(a);
        assert_eq!(r, e);
    }
}
