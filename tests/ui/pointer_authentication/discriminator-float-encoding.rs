//@ add-minicore
//@ revisions: aarch64_linux aarch64_darwin x86_64_linux
//@ [aarch64_linux] compile-flags: --target aarch64-unknown-linux-pauthtest
//@ [aarch64_linux] needs-llvm-components: aarch64
//@ [aarch64_darwin] compile-flags: --target aarch64-apple-darwin
//@ [aarch64_darwin] needs-llvm-components: aarch64
//@ [x86_64_linux] compile-flags: --target x86_64-unknown-linux-gnu
//@ [x86_64_linux] needs-llvm-components: x86
//@ dont-require-annotations: ERROR

// Test for 16 and 128 bit wide floating-point types in the discriminator encoding.
//
// Clang's `encodeTypeForFunctionPointerAuth` uses Itanium type codes:
// * `_Float16` is `DF16_`
// * `f128` is the C type with its representation: `long double` (`e`) on
//   AArch64 Linux, where `long double` is IEEE quad precision, and
//   `__float128` (`g`) elsewhere.

#![crate_type = "lib"]
#![feature(no_core, rustc_attrs, f16, f128)]
#![no_core]
#![allow(improper_ctypes_definitions)]

extern crate minicore;

#[rustc_dump_ptrauth_discriminator(ptrauth_encoding)]
extern "C" fn half(x: f16) -> f16 {
    x
}

#[rustc_dump_ptrauth_discriminator(ptrauth_encoding)]
extern "C" fn quad(x: f128) -> f128 {
    x
}
