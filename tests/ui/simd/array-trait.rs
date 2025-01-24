// Figuring out the size of a vector type that depends on traits doesn't ICE

#![allow(dead_code)]


#![feature(repr_simd, intrinsics, generic_const_exprs)]
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
//~| ERROR SIMD vector element type should be a primitive scalar
//~| ERROR unconstrained generic constant

extern "rust-intrinsic" {
    fn simd_insert<T, E>(x: T, idx: u32, y: E) -> T;
    fn simd_extract<T, E>(x: T, idx: u32) -> E;
}

pub fn main() {
    let mut t = T::<i32x4>([0; 4]);
    unsafe {
        t = simd_insert(t, 3, 3);
        assert_eq!(3, simd_extract(t, 3));
    }
}
