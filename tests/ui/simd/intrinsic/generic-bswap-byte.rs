//@ run-pass
#![feature(repr_simd, platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone)]
struct i8x4([i8; 4]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x4([u8; 4]);

extern "platform-intrinsic" {
    fn simd_bswap<T>(x: T) -> T;
}

fn main() {
    unsafe {
        assert_eq!(simd_bswap(i8x4([0, 1, 2, 3])).0, [0, 1, 2, 3]);
        assert_eq!(simd_bswap(u8x4([0, 1, 2, 3])).0, [0, 1, 2, 3]);
    }
}
