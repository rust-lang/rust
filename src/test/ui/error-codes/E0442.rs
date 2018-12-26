#![feature(repr_simd)]
#![feature(platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
struct i8x16(i8, i8, i8, i8, i8, i8, i8, i8,
             i8, i8, i8, i8, i8, i8, i8, i8);
#[repr(simd)]
struct i32x4(i32, i32, i32, i32);
#[repr(simd)]
struct i64x2(i64, i64);

extern "platform-intrinsic" {
    fn x86_mm_adds_epi16(x: i8x16, y: i32x4) -> i64x2;
    //~^ ERROR E0442
    //~| ERROR E0442
    //~| ERROR E0442
}

fn main() {}
