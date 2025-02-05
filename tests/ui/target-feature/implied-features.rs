//@ only-x86_64
//@ build-pass
#![allow(dead_code)]

#[target_feature(enable = "ssse3")]
fn call_ssse3() {}

#[target_feature(enable = "avx")]
fn call_avx() {}

#[target_feature(enable = "avx2")]
fn test_avx2() {
    call_ssse3();
    call_avx();
}

#[target_feature(enable = "fma")]
fn test_fma() {
    call_ssse3();
    call_avx();
}

fn main() {}
