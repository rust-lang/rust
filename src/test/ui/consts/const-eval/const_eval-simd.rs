// run-pass
// compile-flags: -Zunleash-the-miri-inside-of-you
#![feature(repr_simd)]
#![feature(platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)] struct i8x1(i8);

extern "platform-intrinsic" {
    fn simd_insert<T, U>(x: T, idx: u32, val: U) -> T;
    fn simd_extract<T, U>(x: T, idx: u32) -> U;
}

const fn foo(x: i8x1) -> i8 {
    unsafe { simd_insert(x, 0_u32, 42_i8) }.0
}

fn main() {
    const V: i8x1 = i8x1(13);
    const X: i8 = foo(V);
    const Y: i8 = unsafe { simd_extract(V, 0) };
    assert_eq!(X, 42);
    assert_eq!(Y, 13);
}
