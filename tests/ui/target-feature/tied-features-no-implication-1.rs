//@ revisions: paca pacg
//@ compile-flags: --crate-type=rlib --target=aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64
//@[paca] compile-flags: -Ctarget-feature=+paca
//@[pacg] compile-flags: -Ctarget-feature=+pacg
//@ ignore-backends: gcc
//@ add-minicore
// FIXME(#147881): *disable* the features again for minicore as otherwise that will fail to build.
//@ minicore-compile-flags: -C target-feature=-pacg,-paca
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

// In this test, demonstrate that +paca and +pacg both result in the tied feature error if there
// isn't something causing an error.
// See tied-features-no-implication.rs

#[cfg(target_feature = "pacg")]
pub unsafe fn foo() {
}

//~? ERROR the target features paca, pacg must all be either enabled or disabled together
