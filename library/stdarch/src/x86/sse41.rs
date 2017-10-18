
use std::mem;

#[cfg(test)]
use stdsimd_test::assert_instr;

use v128::*;

#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(pblendvb))]
pub unsafe fn _mm_blendv_epi8(a: i8x16, b: i8x16, mask: i8x16) -> i8x16 {
    pblendvb(a, b, mask)
}

#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(pblendw, imm8=0xF0))]
pub unsafe fn _mm_blend_epi16(a: i16x8, b: i16x8, imm8: u8) -> i16x8 {
    macro_rules! call {
        ($imm8:expr) => { pblendw(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Blend packed double-precision (64-bit) floating-point elements from `a` and `b` using `mask`
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(blendvpd))]
pub unsafe fn _mm_blendv_pd(a: f64x2, b: f64x2, mask: f64x2) -> f64x2 {
    blendvpd(a, b, mask)
}

/// Blend packed single-precision (32-bit) floating-point elements from `a` and `b` using `mask`
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(blendvps))]
pub unsafe fn _mm_blendv_ps(a: f32x4, b: f32x4, mask: f32x4) -> f32x4 {
    blendvps(a, b, mask)
}

/// Blend packed double-precision (64-bit) floating-point elements from `a` and `b` using control mask `imm2`
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(blendpd, imm2=0b10))]
pub unsafe fn _mm_blend_pd(a: f64x2, b: f64x2, imm2: u8) -> f64x2 {
    macro_rules! call {
        ($imm2:expr) => { blendpd(a, b, $imm2) }
    }
    constify_imm2!(imm2, call)
}

/// Blend packed single-precision (32-bit) floating-point elements from `a` and `b` using mask `imm4`
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(blendps, imm4=0b0101))]
pub unsafe fn _mm_blend_ps(a: f32x4, b: f32x4, imm4: u8) -> f32x4 {
    macro_rules! call {
        ($imm4:expr) => { blendps(a, b, $imm4) }
    }
    constify_imm4!(imm4, call)
}

/// Extract a single-precision (32-bit) floating-point element from `a`, selected with `imm8`
#[inline(always)]
#[target_feature = "+sse4.1"]
// TODO: Add test for Windows
#[cfg_attr(all(test, not(windows)), assert_instr(extractps, imm8=0))]
pub unsafe fn _mm_extract_ps(a: f32x4, imm8: u8) -> i32 {
    mem::transmute(a.extract(imm8 as u32 & 0b11))
}

/// Extract an 8-bit integer from `a` selected with `imm8`
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(pextrb, imm8=0))]
pub unsafe fn _mm_extract_epi8(a: i8x16, imm8: u8) -> i8 {
    a.extract((imm8 & 0b1111) as u32)
}

/// Extract an 32-bit integer from `a` selected with `imm8`
#[inline(always)]
#[target_feature = "+sse4.1"]
// TODO: Add test for Windows
#[cfg_attr(all(test, not(windows)), assert_instr(pextrd, imm8=1))]
pub unsafe fn _mm_extract_epi32(a: i32x4, imm8: u8) -> i32 {
    a.extract((imm8 & 0b11) as u32)
}

/// Extract an 64-bit integer from `a` selected with `imm8`
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[target_feature = "+sse4.1"]
// TODO: Add test for Windows
#[cfg_attr(all(test, not(windows)), assert_instr(pextrq, imm8=1))]
pub unsafe fn _mm_extract_epi64(a: i64x2, imm8: u8) -> i64 {
    a.extract((imm8 & 0b1) as u32)
}

/// Select a single value in `a` to store at some position in `b`, 
/// Then zero elements according to `imm8`.
/// 
/// `imm8` specifies which bits from operand `a` will be copied, which bits in the 
/// result they will be copied to, and which bits in the result will be
/// cleared. The following assignments are made:
///
/// * Bits `[7:6]` specify the bits to copy from operand `a`:
///     - `00`: Selects bits `[31:0]` from operand `a`.
///     - `01`: Selects bits `[63:32]` from operand `a`.
///     - `10`: Selects bits `[95:64]` from operand `a`.
///     - `11`: Selects bits `[127:96]` from operand `a`.
///
/// * Bits `[5:4]` specify the bits in the result to which the selected bits
/// from operand `a` are copied:
///     - `00`: Copies the selected bits from `a` to result bits `[31:0]`.
///     - `01`: Copies the selected bits from `a` to result bits `[63:32]`.
///     - `10`: Copies the selected bits from `a` to result bits `[95:64]`.
///     - `11`: Copies the selected bits from `a` to result bits `[127:96]`.
///
/// * Bits `[3:0]`: If any of these bits are set, the corresponding result
/// element is cleared.
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(insertps, imm8=0b1010))]
pub unsafe fn _mm_insert_ps(a: f32x4, b: f32x4, imm8: u8) -> f32x4 {
    macro_rules! call {
        ($imm8:expr) => { insertps(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Return a copy of `a` with the 8-bit integer from `i` inserted at a location specified by `imm8`. 
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(pinsrb, imm8=0))]
pub unsafe fn _mm_insert_epi8(a: i8x16, i: i8, imm8: u8) -> i8x16 {
    a.replace((imm8 & 0b1111) as u32, i)
}

/// Return a copy of `a` with the 32-bit integer from `i` inserted at a location specified by `imm8`. 
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(pinsrd, imm8=0))]
pub unsafe fn _mm_insert_epi32(a: i32x4, i: i32, imm8: u8) -> i32x4 {
    a.replace((imm8 & 0b11) as u32, i)
}

/// Return a copy of `a` with the 64-bit integer from `i` inserted at a location specified by `imm8`. 
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(pinsrq, imm8=0))]
pub unsafe fn _mm_insert_epi64(a: i64x2, i: i64, imm8: u8) -> i64x2 {
    a.replace((imm8 & 0b1) as u32, i)
}

/// Compare packed 8-bit integers in `a` and `b`,87 and return packed maximum values in dst. 
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(pmaxsb, imm8=0))]
pub unsafe fn _mm_max_epi8(a: i8x16, b: i8x16) -> i8x16 {
    pmaxsb(a, b)
}

/// Compare packed unsigned 16-bit integers in `a` and `b`, and return packed maximum.
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(pmaxuw, imm8=0))]
pub unsafe fn _mm_max_epu16(a: u16x8, b: u16x8) -> u16x8 {
    pmaxuw(a, b)
}

// Compare packed 32-bit integers in `a` and `b`, and return packed maximum values.
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(pmaxsd, imm8=0))]
pub unsafe fn _mm_max_epi32(a: i32x4, b: i32x4) -> i32x4 {
    pmaxsd(a, b)
}

// Compare packed unsigned 32-bit integers in `a` and `b`, and return packed maximum values.
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(pmaxud, imm8=0))]
pub unsafe fn _mm_max_epu32(a: u32x4, b: u32x4) -> u32x4 {
    pmaxud(a, b)
}

/// Returns the dot product of two f64x2 vectors.
///
/// `imm8[1:0]` is the broadcast mask, and `imm8[5:4]` is the condition mask.
/// If a condition mask bit is zero, the corresponding multiplication is
/// replaced by a value of `0.0`. If a broadcast mask bit is one, the result of
/// the dot product will be stored in the return value component. Otherwise if
/// the broadcast mask bit is zero then the return component will be zero.
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(dppd, imm8 = 0))]
pub unsafe fn _mm_dp_pd(a: f64x2, b: f64x2, imm8: u8) -> f64x2 {
    macro_rules! call {
        ($imm8:expr) => { dppd(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Returns the dot product of two f32x4 vectors.
///
/// `imm8[3:0]` is the broadcast mask, and `imm8[7:4]` is the condition mask.
/// If a condition mask bit is zero, the corresponding multiplication is
/// replaced by a value of `0.0`. If a broadcast mask bit is one, the result of
/// the dot product will be stored in the return value component. Otherwise if
/// the broadcast mask bit is zero then the return component will be zero.
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(dpps, imm8 = 0))]
pub unsafe fn _mm_dp_ps(a: f32x4, b: f32x4, imm8: u8) -> f32x4 {
    macro_rules! call {
        ($imm8:expr) => { dpps(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

#[allow(improper_ctypes)]
extern {
    #[link_name = "llvm.x86.sse41.pblendvb"]
    fn pblendvb(a: i8x16, b: i8x16, mask: i8x16) -> i8x16;
    #[link_name = "llvm.x86.sse41.blendvpd"]
    fn blendvpd(a: f64x2, b: f64x2, mask: f64x2) -> f64x2;
    #[link_name = "llvm.x86.sse41.blendvps"]
    fn blendvps(a: f32x4, b: f32x4, mask: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse41.blendpd"]
    fn blendpd(a: f64x2, b: f64x2, imm2: u8) -> f64x2;
    #[link_name = "llvm.x86.sse41.blendps"]
    fn blendps(a: f32x4, b: f32x4, imm4: u8) -> f32x4;
    #[link_name = "llvm.x86.sse41.pblendw"]
    fn pblendw(a: i16x8, b: i16x8, imm8: u8) -> i16x8;
    #[link_name = "llvm.x86.sse41.insertps"]
    fn insertps(a: f32x4, b: f32x4, imm8: u8) -> f32x4;
    #[link_name = "llvm.x86.sse41.pmaxsb"]
    fn pmaxsb(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.x86.sse41.pmaxuw"]
    fn pmaxuw(a: u16x8, b: u16x8) -> u16x8;
    #[link_name = "llvm.x86.sse41.pmaxsd"]
    fn pmaxsd(a: i32x4, b: i32x4) -> i32x4;
    #[link_name = "llvm.x86.sse41.pmaxud"]
    fn pmaxud(a: u32x4, b: u32x4) -> u32x4;
    #[link_name = "llvm.x86.sse41.dppd"]
    fn dppd(a: f64x2, b: f64x2, imm8: u8) -> f64x2;
    #[link_name = "llvm.x86.sse41.dpps"]
    fn dpps(a: f32x4, b: f32x4, imm8: u8) -> f32x4;
}

#[cfg(test)]
mod tests {
    use std::mem;

    use stdsimd_test::simd_test;

    use v128::*;
    use x86::sse41;

    #[simd_test = "sse4.1"]
    unsafe fn _mm_blendv_epi8() {
        let a = i8x16::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = i8x16::new(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let mask = i8x16::new(
            0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1);
        let e = i8x16::new(
            0, 17, 2, 19, 4, 21, 6, 23, 8, 25, 10, 27, 12, 29, 14, 31);
        assert_eq!(sse41::_mm_blendv_epi8(a, b, mask), e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_blendv_pd() {
        let a = f64x2::splat(0.0);
        let b = f64x2::splat(1.0);
        let mask = mem::transmute(i64x2::new(0, -1));
        let r = sse41::_mm_blendv_pd(a, b, mask);
        let e = f64x2::new(0.0, 1.0);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_blendv_ps() {
        let a = f32x4::splat(0.0);
        let b = f32x4::splat(1.0);
        let mask = mem::transmute(i32x4::new(0,-1, 0, -1));
        let r = sse41::_mm_blendv_ps(a, b, mask);
        let e = f32x4::new(0.0, 1.0, 0.0, 1.0);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_blend_pd() {
        let a = f64x2::splat(0.0);
        let b = f64x2::splat(1.0);
        let r = sse41::_mm_blend_pd(a, b, 0b10);
        let e = f64x2::new(0.0, 1.0);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_blend_ps() {
        let a = f32x4::splat(0.0);
        let b = f32x4::splat(1.0);
        let r = sse41::_mm_blend_ps(a, b, 0b1010);
        let e = f32x4::new(0.0, 1.0, 0.0, 1.0);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_blend_epi16() {
        let a = i16x8::splat(0);
        let b = i16x8::splat(1);
        let r = sse41::_mm_blend_epi16(a, b, 0b1010_1100);
        let e = i16x8::new(0, 0, 1, 1, 0, 1, 0, 1);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_extract_ps() {
        let a = f32x4::new(0.0, 1.0, 2.0, 3.0);
        let r: f32 = mem::transmute(sse41::_mm_extract_ps(a, 1));
        assert_eq!(r, 1.0);
        let r: f32 = mem::transmute(sse41::_mm_extract_ps(a, 5));
        assert_eq!(r, 1.0);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_extract_epi8() {
        let a = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = sse41::_mm_extract_epi8(a, 1);
        assert_eq!(r, 1);
        let r = sse41::_mm_extract_epi8(a, 17);
        assert_eq!(r, 1);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_extract_epi32() {
        let a = i32x4::new(0, 1, 2, 3);
        let r = sse41::_mm_extract_epi32(a, 1);
        assert_eq!(r, 1);
        let r = sse41::_mm_extract_epi32(a, 5);
        assert_eq!(r, 1);
    }

    #[cfg(target_arch = "x86_64")]
    #[simd_test = "sse4.1"]
    unsafe fn _mm_extract_epi64() {
        let a = i64x2::new(0, 1);
        let r = sse41::_mm_extract_epi64(a, 1);
        assert_eq!(r, 1);
        let r = sse41::_mm_extract_epi64(a, 3);
        assert_eq!(r, 1);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_insert_ps() {
        let a = f32x4::splat(1.0);
        let b = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let r = sse41::_mm_insert_ps(a, b, 0b11_00_1100);
        let e = f32x4::new(4.0, 1.0, 0.0, 0.0);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_insert_epi8() {
        let a = i8x16::splat(0);
        let e = i8x16::splat(0).replace(1, 32);
        let r = sse41::_mm_insert_epi8(a, 32, 1);
        assert_eq!(r, e);
        let r = sse41::_mm_insert_epi8(a, 32, 17);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_insert_epi32() {
        let a = i32x4::splat(0);
        let e = i32x4::splat(0).replace(1, 32);
        let r = sse41::_mm_insert_epi32(a, 32, 1);
        assert_eq!(r, e);
        let r = sse41::_mm_insert_epi32(a, 32, 5);
        assert_eq!(r, e);
    }

    #[cfg(target_arch = "x86_64")]
    #[simd_test = "sse4.1"]
    unsafe fn _mm_insert_epi64() {
        let a = i64x2::splat(0);
        let e = i64x2::splat(0).replace(1, 32);
        let r = sse41::_mm_insert_epi64(a, 32, 1);
        assert_eq!(r, e);
        let r = sse41::_mm_insert_epi64(a, 32, 3);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_max_epi8() {
        let a = i8x16::new(1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32);
        let b = i8x16::new(2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31);
        let r = sse41::_mm_max_epi8(a, b);
        let e = i8x16::new(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_max_epu16() {
        let a = u16x8::new(1, 4, 5, 8, 9, 12, 13, 16);
        let b = u16x8::new(2, 3, 6, 7, 10, 11, 14, 15);
        let r = sse41::_mm_max_epu16(a, b);
        let e = u16x8::new(2, 4, 6, 8, 10, 12, 14, 16);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_max_epi32() {
        let a = i32x4::new(1, 4, 5, 8);
        let b = i32x4::new(2, 3, 6, 7);
        let r = sse41::_mm_max_epi32(a, b);
        let e = i32x4::new(2, 4, 6, 8);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_max_epu32() {
        let a = u32x4::new(1, 4, 5, 8);
        let b = u32x4::new(2, 3, 6, 7);
        let r = sse41::_mm_max_epu32(a, b);
        let e = u32x4::new(2, 4, 6, 8);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_dp_pd() {
        let a = f64x2::new(2.0, 3.0);
        let b = f64x2::new(1.0, 4.0);
        let e = f64x2::new(14.0, 0.0);
        assert_eq!(sse41::_mm_dp_pd(a, b, 0b00110001), e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_dp_ps() {
        let a = f32x4::new(2.0, 3.0, 1.0, 10.0);
        let b = f32x4::new(1.0, 4.0, 0.5, 10.0);
        let e = f32x4::new(14.5, 0.0, 14.5, 0.0);
        assert_eq!(sse41::_mm_dp_ps(a, b, 0b01110101), e);
    }
}
