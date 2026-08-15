//@ revisions: small kernel medium large

//@ needs-llvm-components: x86
//@ compile-flags: --target x86_64-unknown-linux-gnu -Zfunction-return=thunk-extern

//@[small] check-pass
//@[small] compile-flags: -Ccode-model=small

//@[kernel] check-pass
//@[kernel] compile-flags: -Ccode-model=kernel

//@[medium] check-pass
//@[medium] compile-flags: -Ccode-model=medium

//@[large] check-fail
//@[large] compile-flags: -Ccode-model=large

#![feature(no_core)]
#![no_core]
#![no_main]

//[large]~? ERROR `-Zfunction-return=thunk-extern` is only supported on non-large code models
