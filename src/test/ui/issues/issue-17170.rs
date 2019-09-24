// run-pass
#![feature(repr_simd)]

#[repr(simd)]
struct T(f64, f64, f64);

static X: T = T(0.0, 0.0, 0.0);

fn main() {
    let _ = X;
}
