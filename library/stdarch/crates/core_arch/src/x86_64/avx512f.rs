use crate::{
    core_arch::{simd::*, x86::*},
    mem::transmute,
};

/// Sets packed 64-bit integers in `dst` with the supplied values.
///
/// [Intel's documentation]( https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,4909&text=_mm512_set_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_set_epi64(
    e0: i64,
    e1: i64,
    e2: i64,
    e3: i64,
    e4: i64,
    e5: i64,
    e6: i64,
    e7: i64,
) -> __m512i {
    _mm512_setr_epi64(e7, e6, e5, e4, e3, e2, e1, e0)
}

/// Sets packed 64-bit integers in `dst` with the supplied values in
/// reverse order.
///
/// [Intel's documentation]( https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,4909&text=_mm512_set_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_setr_epi64(
    e0: i64,
    e1: i64,
    e2: i64,
    e3: i64,
    e4: i64,
    e5: i64,
    e6: i64,
    e7: i64,
) -> __m512i {
    let r = i64x8::new(e0, e1, e2, e3, e4, e5, e6, e7);
    transmute(r)
}

#[cfg(test)]
mod tests {
    use std;
    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;
    use crate::core_arch::x86_64::*;

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_setzero_pd() {
        assert_eq_m512d(_mm512_setzero_pd(), _mm512_set1_pd(0.));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_set1_pd() {
        let expected = _mm512_set_pd(2., 2., 2., 2., 2., 2., 2., 2.);
        assert_eq_m512d(expected, _mm512_set1_pd(2.));
    }

    unsafe fn test_mm512_set1_epi64() {
        let r = _mm512_set_epi64(2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m512i(r, _mm512_set1_epi64(2));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmplt_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(0., 1., -1., f64::MAX, f64::NAN, f64::MIN, 100., -100.);
        let b = _mm512_set1_pd(-1.);
        let m = _mm512_cmplt_pd_mask(a, b);
        assert_eq!(m, 0b00000101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmplt_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(0., 1., -1., f64::MAX, f64::NAN, f64::MIN, 100., -100.);
        let b = _mm512_set1_pd(-1.);
        let mask = 0b01100110;
        let r = _mm512_mask_cmplt_pd_mask(mask, a, b);
        assert_eq!(r, 0b00000100);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpnlt_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(0., 1., -1., f64::MAX, f64::NAN, f64::MIN, 100., -100.);
        let b = _mm512_set1_pd(-1.);
        assert_eq!(_mm512_cmpnlt_pd_mask(a, b), !_mm512_cmplt_pd_mask(a, b));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpnlt_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(0., 1., -1., f64::MAX, f64::NAN, f64::MIN, 100., -100.);
        let b = _mm512_set1_pd(-1.);
        let mask = 0b01111010;
        assert_eq!(_mm512_mask_cmpnlt_pd_mask(mask, a, b), 0b01111010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmple_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(0., 1., -1., f64::MAX, f64::NAN, f64::MIN, 100., -100.);
        let b = _mm512_set1_pd(-1.);
        assert_eq!(_mm512_cmple_pd_mask(a, b), 0b00100101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmple_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(0., 1., -1., f64::MAX, f64::NAN, f64::MIN, 100., -100.);
        let b = _mm512_set1_pd(-1.);
        let mask = 0b01111010;
        assert_eq!(_mm512_mask_cmple_pd_mask(mask, a, b), 0b00100000);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpnle_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(0., 1., -1., f64::MAX, f64::NAN, f64::MIN, 100., -100.);
        let b = _mm512_set1_pd(-1.);
        let m = _mm512_cmpnle_pd_mask(b, a);
        assert_eq!(m, 0b00001101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpnle_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(0., 1., -1., f64::MAX, f64::NAN, f64::MIN, 100., -100.);
        let b = _mm512_set1_pd(-1.);
        let mask = 0b01100110;
        let r = _mm512_mask_cmpnle_pd_mask(mask, b, a);
        assert_eq!(r, 0b00000100);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpeq_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(0., 1., -1., 13., f64::MAX, f64::MIN, f64::NAN, -100.);
        #[rustfmt::skip]
        let b = _mm512_set_pd(0., 1., 13., 42., f64::MAX, f64::MIN, f64::NAN, -100.);
        let m = _mm512_cmpeq_pd_mask(b, a);
        assert_eq!(m, 0b11001101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpeq_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(0., 1., -1., 13., f64::MAX, f64::MIN, f64::NAN, -100.);
        #[rustfmt::skip]
        let b = _mm512_set_pd(0., 1., 13., 42., f64::MAX, f64::MIN, f64::NAN, -100.);
        let mask = 0b01111010;
        let r = _mm512_mask_cmpeq_pd_mask(mask, b, a);
        assert_eq!(r, 0b01001000);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpneq_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(0., 1., -1., 13., f64::MAX, f64::MIN, f64::NAN, -100.);
        #[rustfmt::skip]
        let b = _mm512_set_pd(0., 1., 13., 42., f64::MAX, f64::MIN, f64::NAN, -100.);
        let m = _mm512_cmpneq_pd_mask(b, a);
        assert_eq!(m, 0b00110010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpneq_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(0., 1., -1., 13., f64::MAX, f64::MIN, f64::NAN, -100.);
        #[rustfmt::skip]
        let b = _mm512_set_pd(0., 1., 13., 42., f64::MAX, f64::MIN, f64::NAN, -100.);
        let mask = 0b01111010;
        let r = _mm512_mask_cmpneq_pd_mask(mask, b, a);
        assert_eq!(r, 0b00110010)
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmp_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(0., 1., -1., 13., f64::MAX, f64::MIN, 100., -100.);
        let b = _mm512_set1_pd(-1.);
        let m = _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ);
        assert_eq!(m, 0b00000101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmp_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(0., 1., -1., 13., f64::MAX, f64::MIN, 100., -100.);
        let b = _mm512_set1_pd(-1.);
        let mask = 0b01100110;
        let r = _mm512_mask_cmp_pd_mask(mask, a, b, _CMP_LT_OQ);
        assert_eq!(r, 0b00000100);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmp_round_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(0., 1., -1., 13., f64::MAX, f64::MIN, 100., -100.);
        let b = _mm512_set1_pd(-1.);
        let m = _mm512_cmp_round_pd_mask(a, b, _CMP_LT_OQ, _MM_FROUND_CUR_DIRECTION);
        assert_eq!(m, 0b00000101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmp_round_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(0., 1., -1., 13., f64::MAX, f64::MIN, 100., -100.);
        let b = _mm512_set1_pd(-1.);
        let mask = 0b01100110;
        let r = _mm512_mask_cmp_round_pd_mask(mask, a, b, _CMP_LT_OQ, _MM_FROUND_CUR_DIRECTION);
        assert_eq!(r, 0b00000100);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpord_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(f64::NAN, f64::MAX, f64::NAN, f64::MIN, f64::NAN, -1., f64::NAN, 0.);
        #[rustfmt::skip]
        let b = _mm512_set_pd(f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::MIN, f64::MAX, -1., 0.);
        let m = _mm512_cmpord_pd_mask(a, b);
        assert_eq!(m, 0b00000101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpord_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(f64::NAN, f64::MAX, f64::NAN, f64::MIN, f64::NAN, -1., f64::NAN, 0.);
        #[rustfmt::skip]
        let b = _mm512_set_pd(f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::MIN, f64::MAX, -1., 0.);
        let mask = 0b11000011;
        let m = _mm512_mask_cmpord_pd_mask(mask, a, b);
        assert_eq!(m, 0b00000001);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpunord_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(f64::NAN, f64::MAX, f64::NAN, f64::MIN, f64::NAN, -1., f64::NAN, 0.);
        #[rustfmt::skip]
        let b = _mm512_set_pd(f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::MIN, f64::MAX, -1., 0.);
        let m = _mm512_cmpunord_pd_mask(a, b);

        assert_eq!(m, 0b11111010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpunord_pd_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_pd(f64::NAN, f64::MAX, f64::NAN, f64::MIN, f64::NAN, -1., f64::NAN, 0.);
        #[rustfmt::skip]
        let b = _mm512_set_pd(f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::MIN, f64::MAX, -1., 0.);
        let mask = 0b00001111;
        let m = _mm512_mask_cmpunord_pd_mask(mask, a, b);
        assert_eq!(m, 0b000001010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmplt_epu64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, u64::MAX as i64, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        let m = _mm512_cmplt_epu64_mask(a, b);
        assert_eq!(m, 0b11001111);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmplt_epu64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, u64::MAX as i64, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        let mask = 0b01111010;
        let r = _mm512_mask_cmplt_epu64_mask(mask, a, b);
        assert_eq!(r, 0b01001010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpgt_epu64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, u64::MAX as i64, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        let m = _mm512_cmpgt_epu64_mask(b, a);
        assert_eq!(m, 0b11001111);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpgt_epu64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, u64::MAX as i64, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        let mask = 0b01111010;
        let r = _mm512_mask_cmpgt_epu64_mask(mask, b, a);
        assert_eq!(r, 0b01001010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmple_epu64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, u64::MAX as i64, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        assert_eq!(
            _mm512_cmple_epu64_mask(a, b),
            !_mm512_cmpgt_epu64_mask(a, b)
        )
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmple_epu64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, u64::MAX as i64, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        let mask = 0b01111010;
        assert_eq!(_mm512_mask_cmple_epu64_mask(mask, a, b), 0b01111010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpge_epu64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, u64::MAX as i64, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        assert_eq!(
            _mm512_cmpge_epu64_mask(a, b),
            !_mm512_cmplt_epu64_mask(a, b)
        );
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpge_epu64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, u64::MAX as i64, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        let mask = 0b01111010;
        assert_eq!(_mm512_mask_cmpge_epu64_mask(mask, a, b), 0b01111010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpeq_epu64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, u64::MAX as i64, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set_epi64(0, 1, 13, 42, i64::MAX, i64::MIN, 100, -100);
        let m = _mm512_cmpeq_epu64_mask(b, a);
        assert_eq!(m, 0b11001111);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpeq_epu64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, u64::MAX as i64, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set_epi64(0, 1, 13, 42, i64::MAX, i64::MIN, 100, -100);
        let mask = 0b01111010;
        let r = _mm512_mask_cmpeq_epu64_mask(mask, b, a);
        assert_eq!(r, 0b01001010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpneq_epu64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, u64::MAX as i64, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set_epi64(0, 1, 13, 42, i64::MAX, i64::MIN, 100, -100);
        let m = _mm512_cmpneq_epu64_mask(b, a);
        assert_eq!(m, !_mm512_cmpeq_epu64_mask(b, a));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpneq_epu64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, u64::MAX as i64, i64::MAX, i64::MIN, -100, 100);
        let b = _mm512_set_epi64(0, 1, 13, 42, i64::MAX, i64::MIN, 100, -100);
        let mask = 0b01111010;
        let r = _mm512_mask_cmpneq_epu64_mask(mask, b, a);
        assert_eq!(r, 0b00110010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmp_epu64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, u64::MAX as i64, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        let m = _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_LT);
        assert_eq!(m, 0b11001111);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmp_epu64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, u64::MAX as i64, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        let mask = 0b01111010;
        let r = _mm512_mask_cmp_epu64_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(r, 0b01001010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmplt_epi64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, 13, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        let m = _mm512_cmplt_epi64_mask(a, b);
        assert_eq!(m, 0b00000101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmplt_epi64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, 13, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        let mask = 0b01100110;
        let r = _mm512_mask_cmplt_epi64_mask(mask, a, b);
        assert_eq!(r, 0b00000100);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpgt_epi64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, 13, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        let m = _mm512_cmpgt_epi64_mask(b, a);
        assert_eq!(m, 0b00000101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpgt_epi64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, 13, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        let mask = 0b01100110;
        let r = _mm512_mask_cmpgt_epi64_mask(mask, b, a);
        assert_eq!(r, 0b00000100);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmple_epi64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, u64::MAX as i64, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        assert_eq!(
            _mm512_cmple_epi64_mask(a, b),
            !_mm512_cmpgt_epi64_mask(a, b)
        )
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmple_epi64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, u64::MAX as i64, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        let mask = 0b01111010;
        assert_eq!(_mm512_mask_cmple_epi64_mask(mask, a, b), 0b00110000);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpge_epi64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, u64::MAX as i64, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        assert_eq!(
            _mm512_cmpge_epi64_mask(a, b),
            !_mm512_cmplt_epi64_mask(a, b)
        )
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpge_epi64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, u64::MAX as i64, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        let mask = 0b01111010;
        assert_eq!(_mm512_mask_cmpge_epi64_mask(mask, a, b), 0b0110000);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpeq_epi64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, 13, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set_epi64(0, 1, 13, 42, i64::MAX, i64::MIN, 100, -100);
        let m = _mm512_cmpeq_epi64_mask(b, a);
        assert_eq!(m, 0b11001111);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpeq_epi64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, 13, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set_epi64(0, 1, 13, 42, i64::MAX, i64::MIN, 100, -100);
        let mask = 0b01111010;
        let r = _mm512_mask_cmpeq_epi64_mask(mask, b, a);
        assert_eq!(r, 0b01001010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_set_epi64() {
        let r = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        assert_eq_m512i(r, _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0))
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_setr_epi64() {
        let r = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        assert_eq_m512i(r, _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0))
    }

    unsafe fn test_mm512_cmpneq_epi64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, 13, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set_epi64(0, 1, 13, 42, i64::MAX, i64::MIN, 100, -100);
        let m = _mm512_cmpneq_epi64_mask(b, a);
        assert_eq!(m, !_mm512_cmpeq_epi64_mask(b, a));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpneq_epi64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, 13, i64::MAX, i64::MIN, -100, 100);
        let b = _mm512_set_epi64(0, 1, 13, 42, i64::MAX, i64::MIN, 100, -100);
        let mask = 0b01111010;
        let r = _mm512_mask_cmpneq_epi64_mask(mask, b, a);
        assert_eq!(r, 0b00110010)
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmp_epi64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, 13, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        let m = _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_LT);
        assert_eq!(m, 0b00000101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmp_epi64_mask() {
        let a = _mm512_set_epi64(0, 1, -1, 13, i64::MAX, i64::MIN, 100, -100);
        let b = _mm512_set1_epi64(-1);
        let mask = 0b01100110;
        let r = _mm512_mask_cmp_epi64_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(r, 0b00000100);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_i32gather_pd() {
        let mut arr = [0f64; 128];
        for i in 0..128 {
            arr[i] = i as f64;
        }
        // A multiplier of 8 is word-addressing
        #[rustfmt::skip]
        let index = _mm256_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112);
        let r = _mm512_i32gather_pd(index, arr.as_ptr() as *const u8, 8);
        assert_eq_m512d(r, _mm512_setr_pd(0., 16., 32., 48., 64., 80., 96., 112.));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_i32gather_pd() {
        let mut arr = [0f64; 128];
        for i in 0..128 {
            arr[i] = i as f64;
        }
        let src = _mm512_set1_pd(2.);
        let mask = 0b10101010;
        #[rustfmt::skip]
        let index = _mm256_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112);
        // A multiplier of 8 is word-addressing
        let r = _mm512_mask_i32gather_pd(src, mask, index, arr.as_ptr() as *const u8, 8);
        assert_eq_m512d(r, _mm512_setr_pd(2., 16., 2., 48., 2., 80., 2., 112.));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_i64gather_pd() {
        let mut arr = [0f64; 128];
        for i in 0..128 {
            arr[i] = i as f64;
        }
        // A multiplier of 8 is word-addressing
        #[rustfmt::skip]
        let index = _mm512_setr_epi64(0, 16, 32, 48, 64, 80, 96, 112);
        let r = _mm512_i64gather_pd(index, arr.as_ptr() as *const u8, 8);
        assert_eq_m512d(r, _mm512_setr_pd(0., 16., 32., 48., 64., 80., 96., 112.));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_i64gather_pd() {
        let mut arr = [0f64; 128];
        for i in 0..128 {
            arr[i] = i as f64;
        }
        let src = _mm512_set1_pd(2.);
        let mask = 0b10101010;
        #[rustfmt::skip]
        let index = _mm512_setr_epi64(0, 16, 32, 48, 64, 80, 96, 112);
        // A multiplier of 8 is word-addressing
        let r = _mm512_mask_i64gather_pd(src, mask, index, arr.as_ptr() as *const u8, 8);
        assert_eq_m512d(r, _mm512_setr_pd(2., 16., 2., 48., 2., 80., 2., 112.));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_i64gather_ps() {
        let mut arr = [0f32; 128];
        for i in 0..128 {
            arr[i] = i as f32;
        }
        // A multiplier of 4 is word-addressing
        #[rustfmt::skip]
        let index = _mm512_setr_epi64(0, 16, 32, 48, 64, 80, 96, 112);
        let r = _mm512_i64gather_ps(index, arr.as_ptr() as *const u8, 4);
        assert_eq_m256(r, _mm256_setr_ps(0., 16., 32., 48., 64., 80., 96., 112.));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_i64gather_ps() {
        let mut arr = [0f32; 128];
        for i in 0..128 {
            arr[i] = i as f32;
        }
        let src = _mm256_set1_ps(2.);
        let mask = 0b10101010;
        #[rustfmt::skip]
        let index = _mm512_setr_epi64(0, 16, 32, 48, 64, 80, 96, 112);
        // A multiplier of 4 is word-addressing
        let r = _mm512_mask_i64gather_ps(src, mask, index, arr.as_ptr() as *const u8, 4);
        assert_eq_m256(r, _mm256_setr_ps(2., 16., 2., 48., 2., 80., 2., 112.));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_i32gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing
        #[rustfmt::skip]
        let index = _mm256_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112);
        let r = _mm512_i32gather_epi64(index, arr.as_ptr() as *const u8, 8);
        assert_eq_m512i(r, _mm512_setr_epi64(0, 16, 32, 48, 64, 80, 96, 112));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_i32gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        let src = _mm512_set1_epi64(2);
        let mask = 0b10101010;
        #[rustfmt::skip]
        let index = _mm256_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112);
        // A multiplier of 8 is word-addressing
        let r = _mm512_mask_i32gather_epi64(src, mask, index, arr.as_ptr() as *const u8, 8);
        assert_eq_m512i(r, _mm512_setr_epi64(2, 16, 2, 48, 2, 80, 2, 112));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_i64gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing
        #[rustfmt::skip]
        let index = _mm512_setr_epi64(0, 16, 32, 48, 64, 80, 96, 112);
        let r = _mm512_i64gather_epi64(index, arr.as_ptr() as *const u8, 8);
        assert_eq_m512i(r, _mm512_setr_epi64(0, 16, 32, 48, 64, 80, 96, 112));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_i64gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        let src = _mm512_set1_epi64(2);
        let mask = 0b10101010;
        #[rustfmt::skip]
        let index = _mm512_setr_epi64(0, 16, 32, 48, 64, 80, 96, 112);
        // A multiplier of 8 is word-addressing
        let r = _mm512_mask_i64gather_epi64(src, mask, index, arr.as_ptr() as *const u8, 8);
        assert_eq_m512i(r, _mm512_setr_epi64(2, 16, 2, 48, 2, 80, 2, 112));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_i64gather_epi32() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing
        #[rustfmt::skip]
        let index = _mm512_setr_epi64(0, 16, 32, 48, 64, 80, 96, 112);
        let r = _mm512_i64gather_epi32(index, arr.as_ptr() as *const u8, 8);
        assert_eq_m256i(r, _mm256_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_i64gather_epi32() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        let src = _mm256_set1_epi32(2);
        let mask = 0b10101010;
        #[rustfmt::skip]
        let index = _mm512_setr_epi64(0, 16, 32, 48, 64, 80, 96, 112);
        // A multiplier of 8 is word-addressing
        let r = _mm512_mask_i64gather_epi32(src, mask, index, arr.as_ptr() as *const u8, 8);
        assert_eq_m256i(r, _mm256_setr_epi32(2, 16, 2, 48, 2, 80, 2, 112));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_i32scatter_pd() {
        let mut arr = [0f64; 128];
        let index = _mm256_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112);
        let src = _mm512_setr_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        // A multiplier of 8 is word-addressing
        _mm512_i32scatter_pd(arr.as_mut_ptr() as *mut u8, index, src, 8);
        let mut expected = [0f64; 128];
        for i in 0..8 {
            expected[i * 16] = (i + 1) as f64;
        }
        assert_eq!(&arr[..], &expected[..],);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_i32scatter_pd() {
        let mut arr = [0f64; 128];
        let mask = 0b10101010;
        let index = _mm256_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112);
        let src = _mm512_setr_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        // A multiplier of 8 is word-addressing
        _mm512_mask_i32scatter_pd(arr.as_mut_ptr() as *mut u8, mask, index, src, 8);
        let mut expected = [0f64; 128];
        for i in 0..4 {
            expected[i * 32 + 16] = 2. * (i + 1) as f64;
        }
        assert_eq!(&arr[..], &expected[..],);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_i64scatter_pd() {
        let mut arr = [0f64; 128];
        let index = _mm512_setr_epi64(0, 16, 32, 48, 64, 80, 96, 112);
        let src = _mm512_setr_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        // A multiplier of 8 is word-addressing
        _mm512_i64scatter_pd(arr.as_mut_ptr() as *mut u8, index, src, 8);
        let mut expected = [0f64; 128];
        for i in 0..8 {
            expected[i * 16] = (i + 1) as f64;
        }
        assert_eq!(&arr[..], &expected[..],);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_i64scatter_pd() {
        let mut arr = [0f64; 128];
        let mask = 0b10101010;
        let index = _mm512_setr_epi64(0, 16, 32, 48, 64, 80, 96, 112);
        let src = _mm512_setr_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        // A multiplier of 8 is word-addressing
        _mm512_mask_i64scatter_pd(arr.as_mut_ptr() as *mut u8, mask, index, src, 8);
        let mut expected = [0f64; 128];
        for i in 0..4 {
            expected[i * 32 + 16] = 2. * (i + 1) as f64;
        }
        assert_eq!(&arr[..], &expected[..],);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_i64scatter_ps() {
        let mut arr = [0f32; 128];
        let index = _mm512_setr_epi64(0, 16, 32, 48, 64, 80, 96, 112);
        let src = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        // A multiplier of 4 is word-addressing
        _mm512_i64scatter_ps(arr.as_mut_ptr() as *mut u8, index, src, 4);
        let mut expected = [0f32; 128];
        for i in 0..8 {
            expected[i * 16] = (i + 1) as f32;
        }
        assert_eq!(&arr[..], &expected[..],);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_i64scatter_ps() {
        let mut arr = [0f32; 128];
        let mask = 0b10101010;
        let index = _mm512_setr_epi64(0, 16, 32, 48, 64, 80, 96, 112);
        let src = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        // A multiplier of 4 is word-addressing
        _mm512_mask_i64scatter_ps(arr.as_mut_ptr() as *mut u8, mask, index, src, 4);
        let mut expected = [0f32; 128];
        for i in 0..4 {
            expected[i * 32 + 16] = 2. * (i + 1) as f32;
        }
        assert_eq!(&arr[..], &expected[..],);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_i32scatter_epi64() {
        let mut arr = [0i64; 128];
        let index = _mm256_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112);
        let src = _mm512_setr_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        // A multiplier of 8 is word-addressing
        _mm512_i32scatter_epi64(arr.as_mut_ptr() as *mut u8, index, src, 8);
        let mut expected = [0i64; 128];
        for i in 0..8 {
            expected[i * 16] = (i + 1) as i64;
        }
        assert_eq!(&arr[..], &expected[..],);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_i32scatter_epi64() {
        let mut arr = [0i64; 128];
        let mask = 0b10101010;
        let index = _mm256_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112);
        let src = _mm512_setr_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        // A multiplier of 8 is word-addressing
        _mm512_mask_i32scatter_epi64(arr.as_mut_ptr() as *mut u8, mask, index, src, 8);
        let mut expected = [0i64; 128];
        for i in 0..4 {
            expected[i * 32 + 16] = 2 * (i + 1) as i64;
        }
        assert_eq!(&arr[..], &expected[..],);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_i64scatter_epi64() {
        let mut arr = [0i64; 128];
        let index = _mm512_setr_epi64(0, 16, 32, 48, 64, 80, 96, 112);
        let src = _mm512_setr_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        // A multiplier of 8 is word-addressing
        _mm512_i64scatter_epi64(arr.as_mut_ptr() as *mut u8, index, src, 8);
        let mut expected = [0i64; 128];
        for i in 0..8 {
            expected[i * 16] = (i + 1) as i64;
        }
        assert_eq!(&arr[..], &expected[..],);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_i64scatter_epi64() {
        let mut arr = [0i64; 128];
        let mask = 0b10101010;
        let index = _mm512_setr_epi64(0, 16, 32, 48, 64, 80, 96, 112);
        let src = _mm512_setr_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        // A multiplier of 8 is word-addressing
        _mm512_mask_i64scatter_epi64(arr.as_mut_ptr() as *mut u8, mask, index, src, 8);
        let mut expected = [0i64; 128];
        for i in 0..4 {
            expected[i * 32 + 16] = 2 * (i + 1) as i64;
        }
        assert_eq!(&arr[..], &expected[..],);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_i64scatter_epi32() {
        let mut arr = [0i32; 128];
        let index = _mm512_setr_epi64(0, 16, 32, 48, 64, 80, 96, 112);
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        // A multiplier of 4 is word-addressing
        _mm512_i64scatter_epi32(arr.as_mut_ptr() as *mut u8, index, src, 4);
        let mut expected = [0i32; 128];
        for i in 0..8 {
            expected[i * 16] = (i + 1) as i32;
        }
        assert_eq!(&arr[..], &expected[..],);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_i64scatter_epi32() {
        let mut arr = [0i32; 128];
        let mask = 0b10101010;
        let index = _mm512_setr_epi64(0, 16, 32, 48, 64, 80, 96, 112);
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        // A multiplier of 4 is word-addressing
        _mm512_mask_i64scatter_epi32(arr.as_mut_ptr() as *mut u8, mask, index, src, 4);
        let mut expected = [0i32; 128];
        for i in 0..4 {
            expected[i * 32 + 16] = 2 * (i + 1) as i32;
        }
        assert_eq!(&arr[..], &expected[..],);
    }
}
