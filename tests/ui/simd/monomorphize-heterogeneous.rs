#![feature(repr_simd)]

#[repr(simd)]
struct I64F64(i64, f64);
//~^ ERROR SIMD vector should be homogeneous

static X: I64F64 = I64F64(1, 2.0);

fn main() {}
