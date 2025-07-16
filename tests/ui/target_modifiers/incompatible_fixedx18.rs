//@ aux-build:fixed_x18.rs
//@ compile-flags: --target aarch64-unknown-none
//@ needs-llvm-components: aarch64

//@ revisions:allow_match allow_mismatch error_generated
//@[allow_match] compile-flags: -Zfixed-x18
//@[allow_mismatch] compile-flags: -Cunsafe-allow-abi-mismatch=fixed-x18
// [error_generated] no extra compile-flags
//@[allow_mismatch] check-pass
//@[allow_match] check-pass

#![feature(no_core)]
//[error_generated]~^ ERROR mixing `-Zfixed-x18` will cause an ABI mismatch in crate `incompatible_fixedx18`
#![crate_type = "rlib"]
#![no_core]

extern crate fixed_x18;
