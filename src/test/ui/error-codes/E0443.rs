#![feature(repr_simd)]
#![feature(platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
struct i16x8(i16, i16, i16, i16, i16, i16, i16, i16);
#[repr(simd)]
struct i64x8(i64, i64, i64, i64, i64, i64, i64, i64);

extern "platform-intrinsic" {
    fn x86_mm_adds_epi16(x: i16x8, y: i16x8) -> i64x8; //~ ERROR E0443
}

fn main() {}
