#![feature(repr_simd)]
#![feature(platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
struct f64x2(f64, f64);

extern "platform-intrinsic" {
    fn x86_mm_movemask_pd(x: f64x2, y: f64x2, z: f64x2) -> i32; //~ ERROR E0444
}

fn main() {}
