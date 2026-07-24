// Tests that default unspecified target modifier value in dependency crate is ok linked
// with the same value, explicitly specified

//@ aux-build:default_reg_struct_return.rs
//@ compile-flags: --target i686-unknown-linux-gnu -Cpanic=abort -Zunstable-options
//@ needs-llvm-components: x86

//@ revisions: ok error_explicit error
// [ok] no extra compile-flags
//@[error_explicit] compile-flags: -Treg-struct-return=false
//@[error] compile-flags: -Treg-struct-return=true
//@[ok] check-pass
//@ ignore-backends: gcc

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]

extern crate default_reg_struct_return;

//[error_explicit]~? ERROR mixing `-Treg-struct-return` will cause an ABI mismatch in crate `defaults_check`
//[error]~? ERROR mixing `-Treg-struct-return` will cause an ABI mismatch in crate `defaults_check`
