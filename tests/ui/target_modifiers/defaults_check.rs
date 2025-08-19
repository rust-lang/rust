// Tests that default unspecified target modifier value in dependency crate is ok linked
// with the same value, explicitly specified

//@ aux-build:default_reg_struct_return.rs
//@ compile-flags: --target i686-unknown-linux-gnu -Cpanic=abort
//@ needs-llvm-components: x86

//@ revisions: ok ok_explicit error
// [ok] no extra compile-flags
//@[ok_explicit] compile-flags: -Zreg-struct-return=false
//@[error] compile-flags: -Zreg-struct-return=true
//@[ok] check-pass
//@[ok_explicit] check-pass

#![feature(no_core)]
//[error]~^ ERROR mixing `-Zreg-struct-return` will cause an ABI mismatch in crate `defaults_check`
#![crate_type = "rlib"]
#![no_core]

extern crate default_reg_struct_return;
