// build-fail
#![allow(non_camel_case_types)]
#![feature(repr_simd, platform_intrinsics)]

// Test for #73542 to verify out-of-bounds shuffle vectors do not compile.

#[repr(simd)]
#[derive(Copy, Clone)]
struct f32x4(f32, f32, f32, f32);

extern "platform-intrinsic" {
    pub fn simd_shuffle4<T, U>(x: T, y: T, idx: [u32; 4]) -> U;
}


fn main() {
    unsafe {
        let vec1 = f32x4(1.0, 2.0, 3.0, 4.0);
        let vec2 = f32x4(10_000.0, 20_000.0, 30_000.0, 40_000.0);
        let shuffled: f32x4 = simd_shuffle4(vec1, vec2, [0, 4, 7, 9]);
        //~^ ERROR: invalid monomorphization of `simd_shuffle4` intrinsic: shuffle index #3 is out
    }
}
