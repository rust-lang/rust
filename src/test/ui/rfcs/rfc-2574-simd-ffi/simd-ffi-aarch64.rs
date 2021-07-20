// only-aarch64

#![feature(repr_simd)]
#![feature(simd_ffi)]
#![allow(non_camel_case_types)]
#![cfg(target_arch = "aarch64")]

#[repr(simd)]
struct v1024(i128, i128, i128, i128, i128, i128, i128, i128);

extern {
    fn foo(x: v1024); //~ ERROR use of SIMD type `v1024` in FFI not supported by any target features
    fn bar(x: i32, y: v1024); //~ ERROR use of SIMD type `v1024` in FFI not supported by any target features
    fn baz(x: i32) -> v1024; //~ ERROR use of SIMD type `v1024` in FFI not supported by any target features

    fn qux_fail(x: v64); //~ ERROR use of SIMD type `v64` in FFI requires `#[target_feature(enable = "neon")]`
    #[target_feature(enable = "neon")]
    fn qux(x: v64);

    fn quux_fail(x: v64i); //~ ERROR use of SIMD type `v64i` in FFI requires `#[target_feature(enable = "neon")]`
    #[target_feature(enable = "neon")]
    fn quux(x: v64i);

    fn quuux_fail(x: v128); //~ ERROR use of SIMD type `v128` in FFI requires `#[target_feature(enable = "neon")]`
    #[target_feature(enable = "neon")]
    fn quuux(x: v128);

    fn quuuux_fail(x: v128i); //~ ERROR use of SIMD type `v128i` in FFI requires `#[target_feature(enable = "neon")]`
    #[target_feature(enable = "neon")]
    fn quuuux(x: v128); //~ ERROR use of SIMD type `v128i` in FFI not supported by any target features
}

fn main() {}

#[repr(simd)]
struct v128(i32, i32, i32, i32);

#[repr(simd)]
struct v64(i32, i32);

#[repr(simd)]
struct v128i(i64, i64);

#[repr(simd)]
struct v64i(i64);
