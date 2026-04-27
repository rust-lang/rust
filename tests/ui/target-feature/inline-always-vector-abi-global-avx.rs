//@ run-pass
//@ compile-flags: -C opt-level=3 -Ctarget-feature=+avx
//@ only-x86_64
//@ only-linux
//@ ignore-backends: gcc

#![feature(target_feature_inline_always)]

use std::arch::x86_64::__m256;

const EXPECTED: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

#[inline(never)]
#[target_feature(enable = "sse")]
fn f(x: &__m256) {
    let x = unsafe { std::mem::transmute::<_, [u32; 8]>(*x) };
    assert_eq!(x, EXPECTED);
}

#[inline(always)]
#[target_feature(enable = "sse")]
fn g(x: &__m256) {
    f(x);
}

#[target_feature(enable = "avx")]
fn h(x: &__m256) {
    g(x);
}

fn main() {
    let x = std::hint::black_box(unsafe { std::mem::transmute::<_, __m256>(EXPECTED) });
    unsafe { h(&x); }
}
