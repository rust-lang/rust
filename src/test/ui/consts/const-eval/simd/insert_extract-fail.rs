#![feature(const_fn)]
#![feature(repr_simd)]
#![feature(platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)] struct i8x1(i8);

extern "platform-intrinsic" {
    fn simd_insert<T, U>(x: T, idx: u32, val: U) -> T;
}

const fn foo(x: i8x1) -> i8 {
    // 42 is a i16 that does not fit in a i8
    unsafe { simd_insert(x, 0_u32, 42_i16) }.0  //~ ERROR
}

fn main() {
    const V: i8x1 = i8x1(13);
    const X: i8 = foo(V);
}
