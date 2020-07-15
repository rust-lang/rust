//! Utilities used in testing the x86 intrinsics

use crate::core_arch::x86::*;

#[target_feature(enable = "mmx")]
pub unsafe fn assert_eq_m64(a: __m64, b: __m64) {
    union A {
        a: __m64,
        b: u64,
    }
    assert_eq!(A { a }.b, A { a: b }.b)
}

#[target_feature(enable = "sse2")]
pub unsafe fn assert_eq_m128i(a: __m128i, b: __m128i) {
    union A {
        a: __m128i,
        b: [u64; 2],
    }
    assert_eq!(A { a }.b, A { a: b }.b)
}

#[target_feature(enable = "sse2")]
pub unsafe fn assert_eq_m128d(a: __m128d, b: __m128d) {
    if _mm_movemask_pd(_mm_cmpeq_pd(a, b)) != 0b11 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[target_feature(enable = "sse2")]
pub unsafe fn get_m128d(a: __m128d, idx: usize) -> f64 {
    union A {
        a: __m128d,
        b: [f64; 2],
    };
    A { a }.b[idx]
}

#[target_feature(enable = "sse")]
pub unsafe fn assert_eq_m128(a: __m128, b: __m128) {
    let r = _mm_cmpeq_ps(a, b);
    if _mm_movemask_ps(r) != 0b1111 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[target_feature(enable = "sse")]
pub unsafe fn get_m128(a: __m128, idx: usize) -> f32 {
    union A {
        a: __m128,
        b: [f32; 4],
    };
    A { a }.b[idx]
}

// not actually an intrinsic but useful in various tests as we proted from
// `i64x2::new` which is backwards from `_mm_set_epi64x`
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_setr_epi64x(a: i64, b: i64) -> __m128i {
    _mm_set_epi64x(b, a)
}

#[target_feature(enable = "avx")]
pub unsafe fn assert_eq_m256i(a: __m256i, b: __m256i) {
    union A {
        a: __m256i,
        b: [u64; 4],
    }
    assert_eq!(A { a }.b, A { a: b }.b)
}

#[target_feature(enable = "avx")]
pub unsafe fn assert_eq_m256d(a: __m256d, b: __m256d) {
    let cmp = _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
    if _mm256_movemask_pd(cmp) != 0b1111 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[target_feature(enable = "avx")]
pub unsafe fn get_m256d(a: __m256d, idx: usize) -> f64 {
    union A {
        a: __m256d,
        b: [f64; 4],
    };
    A { a }.b[idx]
}

#[target_feature(enable = "avx")]
pub unsafe fn assert_eq_m256(a: __m256, b: __m256) {
    let cmp = _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
    if _mm256_movemask_ps(cmp) != 0b11111111 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[target_feature(enable = "avx")]
pub unsafe fn get_m256(a: __m256, idx: usize) -> f32 {
    union A {
        a: __m256,
        b: [f32; 8],
    };
    A { a }.b[idx]
}

// These intrinsics doesn't exist on x86 b/c it requires a 64-bit register,
// which doesn't exist on x86!
#[cfg(target_arch = "x86")]
mod x86_polyfill {
    use crate::core_arch::x86::*;

    pub unsafe fn _mm_insert_epi64(a: __m128i, val: i64, idx: i32) -> __m128i {
        union A {
            a: __m128i,
            b: [i64; 2],
        };
        let mut a = A { a };
        a.b[idx as usize] = val;
        a.a
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn _mm256_insert_epi64(a: __m256i, val: i64, idx: i32) -> __m256i {
        union A {
            a: __m256i,
            b: [i64; 4],
        };
        let mut a = A { a };
        a.b[idx as usize] = val;
        a.a
    }
}
#[cfg(target_arch = "x86_64")]
mod x86_polyfill {
    pub use crate::core_arch::x86_64::{_mm256_insert_epi64, _mm_insert_epi64};
}
pub use self::x86_polyfill::*;

pub unsafe fn assert_eq_m512i(a: __m512i, b: __m512i) {
    union A {
        a: __m512i,
        b: [i32; 16],
    }
    assert_eq!(A { a }.b, A { a: b }.b)
}

pub unsafe fn assert_eq_m512(a: __m512, b: __m512) {
    let cmp = _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);
    if cmp != 0b11111111_11111111 {
        panic!("{:?} != {:?}", a, b);
    }
}

pub unsafe fn assert_eq_m512d(a: __m512d, b: __m512d) {
    let cmp = _mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ);
    if cmp != 0b11111111 {
        panic!("{:?} != {:?}", a, b);
    }
}
