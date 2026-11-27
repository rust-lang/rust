//@ run-pass
//@ ignore-backends: gcc

#![feature(repr_simd)]

#[repr(simd)]
struct T([f64; 3]);

static X: T = T([0.0, 0.0, 0.0]);

fn main() {
    let _ = X;
}
