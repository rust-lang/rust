#![feature(repr_simd)]
#![feature(platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
struct i16x8(i16, i16, i16, i16, i16, i16, i16, i16);

extern "platform-intrinsic" {
    fn x86_mm_adds_ep16(x: i16x8, y: i16x8) -> i16x8; //~ ERROR E0441
}

fn main() {}
