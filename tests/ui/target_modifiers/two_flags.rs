//@ aux-build:wrong_regparm_and_ret.rs
//@ compile-flags: --target i686-unknown-linux-gnu
//@ needs-llvm-components: x86

//@ revisions:two_allowed unknown_allowed
//@[two_allowed] compile-flags: -Cunsafe-allow-abi-mismatch=regparm,reg-struct-return
//@[unknown_allowed] compile-flags: -Cunsafe-allow-abi-mismatch=unknown_flag -Zregparm=2 -Zreg-struct-return=true
//@[two_allowed] check-pass

#![feature(no_core)]
//[unknown_allowed]~^ ERROR unknown target modifier `unknown_flag`, requested by `-Cunsafe-allow-abi-mismatch=unknown_flag`
#![crate_type = "rlib"]
#![no_core]

extern crate wrong_regparm_and_ret;
