//@ aux-build:wrong_regparm.rs
//@ compile-flags: --target i686-unknown-linux-gnu -Tregparm=1 -Zunstable-options
//@ needs-llvm-components: x86

//@ revisions:allow_regparm_mismatch allow_no_value error_generated
//@[allow_regparm_mismatch] compile-flags: -Cunsafe-allow-abi-mismatch=regparm
//@[allow_no_value] compile-flags: -Cunsafe-allow-abi-mismatch
// [error_generated] no extra compile-flags
//@[allow_regparm_mismatch] check-pass
//@ ignore-backends: gcc

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]

extern crate wrong_regparm;

//[allow_no_value]~? ERROR codegen option `unsafe-allow-abi-mismatch` requires a comma-separated list of strings
//[error_generated]~? ERROR mixing `-Tregparm` will cause an ABI mismatch in crate `incompatible_regparm`
