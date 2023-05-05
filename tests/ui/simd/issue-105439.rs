// build-pass

#![crate_type = "lib"]

#![feature(repr_simd)]
#![feature(platform_intrinsics)]

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
#[repr(simd)]
pub struct i32x4([i32; 4]);

pub fn f(a: i32x4) -> [i32; 4] {
    let b = a;
    b.0
}
