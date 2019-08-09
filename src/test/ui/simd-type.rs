#![feature(repr_simd)]
#![allow(non_camel_case_types)]

#[repr(simd)]
struct empty; //~ ERROR SIMD vector cannot be empty

#[repr(simd)]
struct i64f64(i64, f64); //~ ERROR SIMD vector should be homogeneous

fn main() {}
