#![feature(is_subnormal)]

#[macro_use]
mod ops_macros;
impl_float_tests! { SimdF32, f32, i32 }
