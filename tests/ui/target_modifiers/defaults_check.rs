// Tests that default unspecified target modifier value in dependency crate is ok linked
// with the same value, explicitly specified
//@ aux-crate:default_reg_struct_return=default_reg_struct_return.rs
//@ compile-flags: --target i686-unknown-linux-gnu -Cpanic=abort
//@ revisions:error ok ok_explicit
//@[ok] compile-flags:
//@[ok_explicit] compile-flags: -Zreg-struct-return=false
//@[error] compile-flags: -Zreg-struct-return=true
//@ needs-llvm-components: x86
//@[ok] build-pass
//@[ok_explicit] build-pass

#![crate_type = "lib"]
//[error]~^ ERROR mixing `-Zreg-struct-return` will cause an ABI mismatch in crate `defaults_check`
#![no_core]
#![feature(no_core, lang_items, repr_simd)]

fn foo() {
    default_reg_struct_return::somefun();
}
