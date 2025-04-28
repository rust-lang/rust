//@ revisions: paca pacg
//@ compile-flags: --crate-type=rlib --target=aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64
//@[paca] compile-flags: -Ctarget-feature=+paca
//@[pacg] compile-flags: -Ctarget-feature=+pacg

#![feature(no_core, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized {}

// Can't use `compile_error!` here without `core`/`std` but requiring these makes this test only
// work if you have libcore built in the sysroot for `aarch64-unknown-linux-gnu`. Can't run this
// test on any aarch64 platform because they all have different default available features - as
// written, this test depends on `aarch64-unknown-linux-gnu` having -paca,-pacg by default.
// Cause a multiple definition error instead.
fn foo() {}

// Enabling one of the tied features does not imply the other is enabled.
//
// With +paca, this multiple definition doesn't cause an error because +paca hasn't implied
// +pacg. With +pacg, the multiple definition error is emitted (and the tied feature error would
// be).

#[cfg(target_feature = "pacg")]
pub unsafe fn foo() {} //[pacg]~ ERROR the name `foo` is defined multiple times

//[paca]~? ERROR the target features paca, pacg must all be either enabled or disabled together
