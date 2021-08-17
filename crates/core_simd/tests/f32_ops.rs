#![feature(portable_simd)]

#[macro_use]
mod ops_macros;
impl_float_tests! { f32, i32 }
