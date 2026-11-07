// Test that we do not get an ABI mismatch error when a default sanitizer is not
// explicitly provided via a -Zsanitizer flag.
//
// riscv64gc-unknown-fuchsia has shadow-call-stack as a default sanitizer.
// Compiling one crate without `-Zsanitizer` and another crate with the target's
// default sanitizer explicitly specified (-Zsanitizer=shadow-call-stack)
// must be accepted.

//@ aux-build:sanitizer_default_implicit.rs
//@ aux-build:sanitizer_default_explicit.rs
//@ aux-build:sanitizer_non_default.rs
//@ compile-flags: --target riscv64gc-unknown-fuchsia
//@ needs-llvm-components: riscv
//@ ignore-backends: gcc

//@ revisions: implicit_default explicit_default implicit_mismatch explicit_mismatch
//@[implicit_default] check-pass
//@[explicit_default] compile-flags: -Zsanitizer=shadow-call-stack
//@[explicit_default] check-pass
//@[explicit_mismatch] compile-flags: -Zsanitizer=shadow-call-stack

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]

#[cfg(any(implicit_default, explicit_default))]
extern crate sanitizer_default_implicit;

#[cfg(any(implicit_default, explicit_default))]
extern crate sanitizer_default_explicit;

// We still expect the normal mismatch error with a non-default sanitizer.
#[cfg(any(implicit_mismatch, explicit_mismatch))]
extern crate sanitizer_non_default;
//[implicit_mismatch]~? ERROR mixing `-Zsanitizer` will cause an ABI mismatch in crate `sanitizer_default`
//[explicit_mismatch]~? ERROR mixing `-Zsanitizer` will cause an ABI mismatch in crate `sanitizer_default`
