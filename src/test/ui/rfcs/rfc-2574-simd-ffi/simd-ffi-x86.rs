// only-x86_64

#![feature(repr_simd)]
#![feature(simd_ffi)]
#![feature(avx512_target_feature)]
#![allow(non_camel_case_types, improper_ctypes)]
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

#[repr(simd)]
struct v1024(i128, i128, i128, i128, i128, i128, i128, i128);

extern {
    fn foo(x: v1024); //~ ERROR use of SIMD type `v1024` in FFI not supported by any target features
    fn bar(x: i32, y: v1024); //~ ERROR use of SIMD type `v1024` in FFI not supported by any target features
    fn baz(x: i32) -> v1024; //~ ERROR use of SIMD type `v1024` in FFI not supported by any target features

    fn qux_fail(x: v128); //~ ERROR use of SIMD type `v128` in FFI requires `#[target_feature(enable = "sse")]`
    #[target_feature(enable = "sse")]
    fn qux(x: v128);
    #[target_feature(enable = "sse4.2")]
    fn qux2(x: v128);
    #[target_feature(enable = "ssse3")]
    fn qux3(x: v128);
    #[target_feature(enable = "avx")]
    fn qux4(x: v128);
    #[target_feature(enable = "avx2")]
    fn qux5(x: v128);
    #[target_feature(enable = "avx512f")]
    fn qux6(x: v128);

    fn quux_fail(x: v256); //~ ERROR use of SIMD type `v256` in FFI requires `#[target_feature(enable = "avx")]`
    #[target_feature(enable = "sse4.2")]
    fn quux_fail2(x: v256); //~ ERROR use of SIMD type `v256` in FFI requires `#[target_feature(enable = "avx")]`
    #[target_feature(enable = "avx")]
    fn quux(x: v256);
    #[target_feature(enable = "avx2")]
    fn quux2(x: v256);
    #[target_feature(enable = "avx512f")]
    fn quux3(x: v256);

    fn quuux_fail(x: v512); //~ ERROR use of SIMD type `v512` in FFI requires `#[target_feature(enable = "avx512f")]`
    #[target_feature(enable = "sse")]
    fn quuux_fail2(x: v512); //~ ERROR use of SIMD type `v512` in FFI requires `#[target_feature(enable = "avx512f")]`
    #[target_feature(enable = "avx2")]
    fn quuux_fail3(x: v512); //~ ERROR use of SIMD type `v512` in FFI requires `#[target_feature(enable = "avx512f")]`
    #[target_feature(enable = "avx512f")]
    fn quuux(x: v512);
}

fn main() {}

#[repr(simd)]
struct v128(i128);

#[repr(simd)]
struct v256(i128, i128);

#[repr(simd)]
struct v512(i128, i128, i128, i128);
