use std::mem::transmute;

use super::{__m128d, __m128i, f64x2, u8x16};
use {simd_add, simd_extract, simd_insert};

pub unsafe fn _mm_add_epi8(a: __m128i, b: __m128i) -> __m128i {
    transmute(simd_add::<u8x16>(transmute(a), transmute(b)))
}

pub unsafe fn _mm_add_sd(a: __m128d, b: __m128d) -> __m128d {
    let alow = simd_extract::<f64x2, f64>(transmute(a), 0);
    let blow = simd_extract::<f64x2, f64>(transmute(b), 0);
    transmute(simd_insert::<f64x2, f64>(transmute(a), 0, alow + blow))
}

pub unsafe fn _mm_add_pd(a: __m128d, b: __m128d) -> __m128d {
    transmute(simd_add::<f64x2>(transmute(a), transmute(b)))
}

pub unsafe fn _mm_load_pd(mem_addr: *const f64) -> __m128d {
    *(mem_addr as *const __m128d)
}

pub unsafe fn _mm_store_pd(mem_addr: *mut f64, a: __m128d) {
    *(mem_addr as *mut __m128d) = a;
}
