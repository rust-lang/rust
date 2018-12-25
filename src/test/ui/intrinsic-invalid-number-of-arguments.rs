// Test number of arguments in platform-specific intrinsic function
// This is the error E0444

#![feature(repr_simd, platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
struct f64x2(f64, f64);

extern "platform-intrinsic" {
    fn x86_mm_movemask_pd(x: f64x2, y: f64x2, z: f64x2) -> i32; //~ platform-specific intrinsic
}

pub fn main() {
}
