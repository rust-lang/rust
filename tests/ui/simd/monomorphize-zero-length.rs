//@ build-fail

#![feature(repr_simd)]

#[repr(simd)]
struct Simd<T, const N: usize>([T; N]);

fn main() {
    let _empty = Simd([1.0; 0]);
}

//~? ERROR monomorphising SIMD type `Simd<f64, 0>` of zero length
