//@ run-pass
//@ only-x86_64
//@ compile-flags: -C opt-level=3

// Regression test for issue #79865.
// The assertion will fail when compiled with Rust 1.56..=1.59
// due to an LLVM miscompilation.

use std::arch::x86_64::*;

fn main() {
    if is_x86_feature_detected!("avx") {
        let res: [f64; 4] = unsafe { std::mem::transmute::<_, _>(first()) };
        assert_eq!(res, [22.0, 44.0, 66.0, 88.0]);
    }
}

#[target_feature(enable = "avx")]
unsafe fn first() -> __m256d {
    second()
}

unsafe fn second() -> __m256d {
    let v0 = _mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
    let v1 = _mm256_setr_pd(10.0, 20.0, 30.0, 40.0);

    // needs to be called twice to hit the miscompilation
    let (add, _) = add_sub(v0, v1);
    let (add, _) = add_sub(add, add);
    add
}

#[inline(never)] // needed to hit the miscompilation
unsafe fn add_sub(v1: __m256d, v0: __m256d) -> (__m256d, __m256d) {
    let add = _mm256_add_pd(v0, v1);
    let sub = _mm256_sub_pd(v0, v1);
    (add, sub)
}
