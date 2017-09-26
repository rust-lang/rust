use v128::*;
use x86::__m128i;

#[cfg(test)]
use stdsimd_test::assert_instr;

#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(pblendvb))]
pub fn _mm_blendv_epi8(
    a: __m128i,
    b: __m128i,
    mask: __m128i,
) -> __m128i {
    unsafe { pblendvb(a, b, mask) }
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
pub fn _mm_dp_pd(a: f64x2, b: f64x2, imm8: u8) -> f64x2 {
    macro_rules! call {
        ($imm8:expr) => {
            unsafe { dppd(a, b, $imm8) }
        }
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
pub fn _mm_dp_ps(a: f32x4, b: f32x4, imm8: u8) -> f32x4 {
    macro_rules! call {
        ($imm8:expr) => {
            unsafe { dpps(a, b, $imm8) }
        }
    }
    constify_imm8!(imm8, call)
}

#[allow(improper_ctypes)]
extern {
    #[link_name = "llvm.x86.sse41.pblendvb"]
    fn pblendvb(a: __m128i, b: __m128i, mask: __m128i) -> __m128i;
    #[link_name = "llvm.x86.sse41.dppd"]
    fn dppd(a: f64x2, b: f64x2, imm8: u8) -> f64x2;
    #[link_name = "llvm.x86.sse41.dpps"]
    fn dpps(a: f32x4, b: f32x4, imm8: u8) -> f32x4;
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use v128::*;
    use x86::sse41;

    #[simd_test = "sse4.1"]
    fn _mm_blendv_epi8() {
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
    fn _mm_dp_pd() {
        let a = f64x2::new(2.0, 3.0);
        let b = f64x2::new(1.0, 4.0);
        let e = f64x2::new(14.0, 0.0);
        assert_eq!(sse41::_mm_dp_pd(a, b, 0b00110001), e);
    }

    #[simd_test = "sse4.1"]
    fn _mm_dp_ps() {
        let a = f32x4::new(2.0, 3.0, 1.0, 10.0);
        let b = f32x4::new(1.0, 4.0, 0.5, 10.0);
        let e = f32x4::new(14.5, 0.0, 14.5, 0.0);
        assert_eq!(sse41::_mm_dp_ps(a, b, 0b01110101), e);
    }
}
