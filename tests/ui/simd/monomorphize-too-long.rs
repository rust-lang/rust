//@ build-fail
//@ error-pattern: monomorphising SIMD type `Simd<u16, 54321>` of length greater than 32768

#![feature(repr_simd)]

#[repr(simd)]
struct Simd<T, const N: usize>([T; N]);

fn main() {
    let _too_big = Simd([1_u16; 54321]);
}
