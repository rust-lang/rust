// only-x86_64

#![feature(repr_simd)]
#![feature(avx512_target_feature)]
#![allow(non_camel_case_types)]
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

#[repr(simd)]
struct v128(i128);

extern {
    fn foo(x: v128); //~ ERROR use of SIMD type `v128` in FFI is unstable
    fn bar(x: i32, y: v128); //~ ERROR use of SIMD type `v128` in FFI is unstable
    fn baz(x: i32) -> v128; //~ ERROR use of SIMD type `v128` in FFI is unstable
}

fn main() {}
