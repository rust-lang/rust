#[macro_use]
#[path = "ops_macros.rs"]
mod macros;
impl_float_tests! { SimdF64, f64, i64 }
