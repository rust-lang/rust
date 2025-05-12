//@only-target: x86_64
#![allow(improper_ctypes_definitions)]
use std::arch::x86_64::*;
use std::mem::transmute;

#[no_mangle]
#[target_feature(enable = "avx")]
pub unsafe extern "C" fn foo(_y: f32, x: __m256) -> __m256 {
    x
}

pub fn bar(x: __m256) -> __m256 {
    // The first and second argument get mixed up here since caller
    // and callee do not have the same feature flags.
    // In Miri, we don't have a concept of "dynamically available feature flags",
    // so this will always lead to an error due to calling a function that requires
    // an unavailable feature. If we ever support dynamically available features,
    // this will need some dedicated checks.
    unsafe { foo(0.0, x) } //~ERROR: unavailable target features
}

fn assert_eq_m256(a: __m256, b: __m256) {
    unsafe { assert_eq!(transmute::<_, [f32; 8]>(a), transmute::<_, [f32; 8]>(b)) }
}

fn main() {
    let input = unsafe { transmute::<_, __m256>([1.0f32; 8]) };
    let copy = bar(input);
    assert_eq_m256(input, copy);
}
