//@ build-fail

#![feature(repr_simd)]

#[repr(simd)]
struct Simd<T, const N: usize>([T; N]);

fn main() {
    let _too_big = Simd([1_u16; 54321]); //~ ERROR the SIMD type `Simd<u16, 54321>` has more elements than the limit 32768
}
