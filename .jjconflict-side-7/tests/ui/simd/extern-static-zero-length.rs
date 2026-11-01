#![feature(repr_simd)]

#[repr(simd)]
struct Simd<T, const N: usize>([T; N]);

unsafe extern "C" {
    static VAR: Simd<u8, 0>;
    //~^ ERROR the SIMD type `Simd<u8, 0>` has zero elements
    static VAR2: Simd<u8, 1_000_000>;
    //~^ ERROR the SIMD type `Simd<u8, 1000000>` has more elements than the limit 32768
}

fn main() {}
