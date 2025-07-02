//@ aux-build:min_function_alignment.rs
//@ compile-flags: --target aarch64-unknown-none
//@ needs-llvm-components: aarch64

//@ revisions:allow_match allow_mismatch error_mismatch error_omitted
//@[allow_match] compile-flags: -Zmin-function-alignment=32
//@[allow_mismatch] compile-flags: -Cunsafe-allow-abi-mismatch=min-function-alignment
//@[error_mismatch] compile-flags: -Zmin-function-alignment=4
//@[error_omitted] compile-flags:
//@[allow_mismatch] check-pass
//@[allow_match] check-pass

#![feature(no_core)]
//[error_mismatch,error_omitted]~^ ERROR mixing `-Zmin-function-alignment` will cause an ABI mismatch in crate `incompatible_min_function_alignment`
#![crate_type = "rlib"]
#![no_core]

extern crate min_function_alignment;
