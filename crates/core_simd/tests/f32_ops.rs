#[macro_use]
#[path = "ops_macros.rs"]
mod macros;
impl_float_tests! { SimdF32, f32, i32 }
