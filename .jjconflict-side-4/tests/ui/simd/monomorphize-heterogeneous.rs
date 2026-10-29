#![feature(repr_simd)]

#[repr(simd)]
struct I64F64(i64, f64);
//~^ ERROR SIMD vector's only field must be an array

#[repr(simd)]
struct I64x4F64x0([i64; 4], [f64; 0]);
//~^ ERROR SIMD vector cannot have multiple fields

static X: I64F64 = I64F64(1, 2.0);

fn main() {}
