// Figuring out the size of a vector type that depends on traits doesn't ICE

#![allow(dead_code)]

// pretty-expanded FIXME #23616

#![feature(repr_simd, platform_intrinsics, generic_const_exprs)]
#![allow(non_camel_case_types, incomplete_features)]

pub trait Simd {
    type Lane: Clone + Copy;
    const SIZE: usize;
}

pub struct i32x4;
impl Simd for i32x4 {
    type Lane = i32;
    const SIZE: usize = 4;
}

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct T<S: Simd>([S::Lane; S::SIZE]);
//~^ ERROR unconstrained generic constant

extern "platform-intrinsic" {
    fn simd_insert<T, E>(x: T, idx: u32, y: E) -> T;
    fn simd_extract<T, E>(x: T, idx: u32) -> E;
}

pub fn main() {
    let mut t = T::<i32x4>([0; 4]);
    unsafe {
        for i in 0_i32..4 {
            t = simd_insert(t, i as u32, i);
        }
        for i in 0_i32..4 {
            assert_eq!(i, simd_extract(t, i as u32));
        }
    }
}
