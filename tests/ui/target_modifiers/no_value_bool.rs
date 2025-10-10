// Tests that bool target modifier value (true) in dependency crate is ok linked
// with the -Zflag specified without value (-Zflag=true is consistent with -Zflag)

//@ aux-build:enabled_reg_struct_return.rs
//@ compile-flags: --target i686-unknown-linux-gnu -Cpanic=abort
//@ needs-llvm-components: x86

//@ revisions: ok ok_explicit error error_explicit
//@[ok] compile-flags: -Zreg-struct-return
//@[ok_explicit] compile-flags: -Zreg-struct-return=true
// [error] no extra compile-flags
//@[error_explicit] compile-flags: -Zreg-struct-return=false
//@[ok] check-pass
//@[ok_explicit] check-pass

#![feature(no_core)]
//[error]~^ ERROR mixing `-Zreg-struct-return` will cause an ABI mismatch in crate `no_value_bool`
//[error_explicit]~^^ ERROR mixing `-Zreg-struct-return` will cause an ABI mismatch in crate `no_value_bool`
#![crate_type = "rlib"]
#![no_core]

extern crate enabled_reg_struct_return;
