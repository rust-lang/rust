use std::arch::x86_64::*;

fn main() {
    if !is_x86_feature_detected!("avx") {
        return println!("AVX is not supported on this machine/build.");
    }
    unsafe {
        let f = _mm256_set_pd(2.0, 2.0, 2.0, 2.0);
        let r = _mm256_mul_pd(f, f);

        union A { a: __m256d, b: [f64; 4] }
        assert_eq!(A { a: r }.b, [4.0, 4.0, 4.0, 4.0]);
    }
}
