//@ add-minicore
//@ revisions: aarch64_linux aarch64_darwin x86_64_linux
//@ [aarch64_linux] compile-flags: --target aarch64-unknown-linux-pauthtest
//@ [aarch64_linux] needs-llvm-components: aarch64
//@ [aarch64_darwin] compile-flags: --target aarch64-apple-darwin
//@ [aarch64_darwin] needs-llvm-components: aarch64
//@ [x86_64_linux] compile-flags: --target x86_64-unknown-linux-gnu
//@ [x86_64_linux] needs-llvm-components: x86
//@ dont-require-annotations: ERROR

// Test for `core::num::Complex<T>` in the discriminator encoding.
//
// Clang's `encodeTypeForFunctionPointerAuth` encodes C `_Complex T` as `C` followed by the encoding
// of `T`. `Complex<f128>` follows the rule from `discriminator-float-encoding.rs`: `Ce` on AArch64
// Linux, `Cg` elsewhere.

#![crate_type = "lib"]
#![feature(no_core, rustc_attrs, f16, f128)]
#![no_core]
#![allow(improper_ctypes_definitions)]

extern crate minicore;
use minicore::num::Complex;

// A `repr(transparent)` wrapper encodes as its field.
#[repr(transparent)]
pub struct Wrap(Complex<f32>);

// An ordinary `repr(C)` struct that happens to be named `Complex`: not `_Complex`.
pub mod user {
    #[repr(C)]
    pub struct Complex {
        pub re: f32,
        pub im: f32,
    }
}

// C: void (*)(float _Complex)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding)]
extern "C" fn cf(_: Complex<f32>) {}

// C: void (*)(double _Complex)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding)]
extern "C" fn cd(_: Complex<f64>) {}

// C: void (*)(int _Complex)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding)]
extern "C" fn ci(_: Complex<i32>) {}

// C: _Float16 _Complex (*)(void)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding)]
extern "C" fn ret_ch() -> Complex<f16> {
    loop {}
}

// C: long double _Complex (*)(void) on AArch64 Linux, __float128 _Complex (*)(void) elsewhere
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding)]
extern "C" fn ret_cq() -> Complex<f128> {
    loop {}
}

// C: double _Complex (*)(float _Complex)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding)]
extern "C" fn cf_to_cd(_: Complex<f32>) -> Complex<f64> {
    loop {}
}

// C: void (*)(float _Complex, int _Complex, double _Complex)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding)]
extern "C" fn mixed(_: Complex<f32>, _: Complex<i32>, _: Complex<f64>) {}

// Same as `cf`.
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding)]
extern "C" fn wrap(_: Wrap) {}

// C: struct Complex { float re, im; }; void (*)(struct Complex)
#[rustc_dump_ptrauth_discriminator(ptrauth_encoding)]
extern "C" fn user_complex(_: user::Complex) {}
