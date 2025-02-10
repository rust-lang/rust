// Tests that default unspecified target modifier value in dependency crate is ok linked
// with the same value, explicitly specified
//@ aux-crate:default_reg_struct_return=default_reg_struct_return.rs
//@ compile-flags: --target i686-unknown-linux-gnu -Cpanic=abort
//@ revisions:error ok ok_explicit
//@[ok] compile-flags:
//@[ok_explicit] compile-flags: -Zreg-struct-return=false
//@[error] compile-flags: -Zreg-struct-return=true

// Auxiliary build problems with aarch64-apple:
// Shared library linking cc seems to convert "-m32" flag into -arch armv4t
// Auxiliary build problems with i686-mingw: linker `cc` not found
//@ only-x86
//@ ignore-windows
//@ ignore-apple
//@ needs-llvm-components: x86
//@[ok] build-pass
//@[ok_explicit] build-pass

#![crate_type = "rlib"]
//[error]~^ ERROR mixing `-Zreg-struct-return` will cause an ABI mismatch in crate `defaults_check`
#![no_core]
#![feature(no_core, lang_items, repr_simd)]

fn foo() {
    default_reg_struct_return::somefun();
}
