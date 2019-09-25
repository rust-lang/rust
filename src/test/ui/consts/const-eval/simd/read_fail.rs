#![feature(const_fn)]
#![feature(platform_intrinsics)]
#![allow(non_camel_case_types)]

extern "platform-intrinsic" {
    fn simd_extract<T, U>(x: T, idx: u32) -> U;
}

const fn foo(x: i8) -> i8 {
    // i8 is not a vector type:
    unsafe { simd_extract(x, 0_u32) }  //~ ERROR
}

fn main() {
    const V: i8 = 13;
    const X: i8 = foo(V);
}
