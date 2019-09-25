// failure-status: 101
// rustc-env:RUST_BACKTRACE=0
#![feature(const_fn)]
#![feature(repr_simd)]
#![feature(platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)] struct i8x1(i8);

extern "platform-intrinsic" {
    fn simd_extract<T, U>(x: T, idx: u32) -> U;
}

const X: i8x1 = i8x1(42);

const fn extract_wrong_vec() -> i8 {
    unsafe { simd_extract(42_i8, 0_u32) }
}

const B: i8 = extract_wrong_vec();

fn main() {}
