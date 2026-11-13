//@ build-pass
//@ revisions: opt0 opt3 asm_opt0 asm_opt3
//@ compile-flags: -Zvirtual-function-elimination -Clto=fat -Zverify-llvm-ir
//@[opt0] compile-flags: -Copt-level=0
//@[opt3] compile-flags: -Copt-level=3
//@[asm_opt0] compile-flags: -Copt-level=0 --emit=asm
//@[asm_opt3] compile-flags: -Copt-level=3 --emit=asm
//@ ignore-backends: gcc
#![crate_type = "rlib"]

// Regression test for https://github.com/rust-lang/rust/issues/124093.
//
// Previously this code hit an error where `llvm.type.checked.load` was still present in the IR.
// This intrinsic is only used for LTO, and not supported for non-LTO compilations, like here
// where we're building an rlib, or on godbolt where `--emit=asm` is passed.

pub trait Trait {
    fn method(&self) -> u32;
}

pub fn call(x: &dyn Trait) -> u32 {
    x.method()
}
