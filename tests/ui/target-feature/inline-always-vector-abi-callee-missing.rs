//@ run-pass
//@ compile-flags: -C opt-level=3
//@ only-x86_64
//@ only-linux
//@ ignore-backends: gcc

#![feature(target_feature_inline_always)]

use std::arch::x86_64::__m256;

#[inline(never)]
#[target_feature(enable = "sse")]
fn f(x: &__m256) {
    let x = unsafe { std::mem::transmute::<_, [u32; 8]>(*x) };
    assert_eq!(x, [1, 2, 3, 4, 5, 6, 7, 8]);
}

#[inline(always)]
#[target_feature(enable = "sse")]
fn g(x: &__m256, y: bool) {
    if y {
        g(x, y);
    } else {
        f(x);
    }
}

#[target_feature(enable = "avx")]
fn h(x: &__m256, y: bool) {
    g(x, y)
    //~^ WARNING call to `#[inline(always)]`-annotated `g` requires the same target features to be inlined [inline_always_mismatching_target_features]
}

fn main() {
    if !is_x86_feature_detected!("avx") {
        return;
    }

    let x = std::hint::black_box(unsafe {
        std::mem::transmute::<_, __m256>([1_u32, 2, 3, 4, 5, 6, 7, 8])
    });
    let y = std::hint::black_box(false);
    unsafe { h(&x, y) }
}
