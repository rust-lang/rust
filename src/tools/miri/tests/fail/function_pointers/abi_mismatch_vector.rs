#![feature(portable_simd)]
use std::simd;

fn main() {
    fn f(_: simd::u32x8) {}

    // These two vector types have the same size but are still not compatible.
    let g = unsafe { std::mem::transmute::<fn(simd::u32x8), fn(simd::u64x4)>(f) };

    g(Default::default()) //~ ERROR: type std::simd::Simd<u32, 8> passing argument of type std::simd::Simd<u64, 4>
}
