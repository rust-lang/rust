//@ aux-build:mips_nan2008.rs
//@ compile-flags: --target mips-unknown-linux-gnu
//@ needs-llvm-components: mips

//@ revisions:allow_match allow_mismatch error_generated
//@[allow_match] compile-flags: -Zmips-nan2008
//@[allow_mismatch] compile-flags: -Cunsafe-allow-abi-mismatch=mips-nan2008
// [error_generated] no extra compile-flags
//@[allow_mismatch] check-pass
//@[allow_match] check-pass
//@ ignore-backends: gcc

#![feature(no_core)]
//[error_generated]~^ ERROR mixing `-Zmips-nan2008` will cause an ABI mismatch in crate `incompatible_mips_nan2008`
#![crate_type = "rlib"]
#![no_core]

extern crate mips_nan2008;
