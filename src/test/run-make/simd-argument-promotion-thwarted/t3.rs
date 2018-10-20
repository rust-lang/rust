use std::arch::x86_64::*;

#[target_feature(enable = "avx")]
unsafe fn avx_mul(a: __m256, b: __m256) -> __m256 {
    _mm256_mul_ps(a, b)
}

#[target_feature(enable = "avx")]
unsafe fn avx_store(p: *mut f32, a: __m256) {
    _mm256_storeu_ps(p, a)
}

#[target_feature(enable = "avx")]
unsafe fn avx_setr(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> __m256 {
    _mm256_setr_ps(a, b, c, d, e, f, g, h)
}

#[target_feature(enable = "avx")]
unsafe fn avx_set1(a: f32) -> __m256 {
    _mm256_set1_ps(a)
}

struct Avx(__m256);

fn mul(a: Avx, b: Avx) -> Avx {
    unsafe { Avx(avx_mul(a.0, b.0)) }
}

fn set1(a: f32) -> Avx {
    unsafe { Avx(avx_set1(a)) }
}

fn setr(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> Avx {
    unsafe { Avx(avx_setr(a, b, c, d, e, f, g, h)) }
}

unsafe fn store(p: *mut f32, a: Avx) {
    avx_store(p, a.0);
}

fn main() {
    if !is_x86_feature_detected!("avx") {
        return println!("AVX is not supported on this machine/build.");
    }
    let mut result = [0.0f32; 8];
    let a = mul(setr(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0), set1(0.25));
    unsafe {
        store(result.as_mut_ptr(), a);
    }

    assert_eq!(result, [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75]);
}
