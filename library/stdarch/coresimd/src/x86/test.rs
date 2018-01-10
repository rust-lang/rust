//! Utilities used in testing the x86 intrinsics

use std::mem;

use x86::*;

#[target_feature = "+sse2"]
pub unsafe fn assert_eq_m128d(a: __m128d, b: __m128d) {
    if _mm_movemask_pd(_mm_cmpeq_pd(a, b)) != 0b11 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[target_feature = "+sse2"]
pub unsafe fn get_m128d(a: __m128d, idx: usize) -> f64 {
    union A { a: __m128d, b: [f64; 2] };
    mem::transmute::<__m128d, A>(a).b[idx]
}

#[target_feature = "+sse"]
pub unsafe fn assert_eq_m128(a: __m128, b: __m128) {
    let r = _mm_cmpeq_ps(a, b);
    if _mm_movemask_ps(r) != 0b1111 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[target_feature = "+sse"]
pub unsafe fn get_m128(a: __m128, idx: usize) -> f32 {
    union A { a: __m128, b: [f32; 4] };
    mem::transmute::<__m128, A>(a).b[idx]
}
