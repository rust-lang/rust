//@ add-core-stubs
//@ revisions: WINDOWS_ ANDROID_
//@ compile-flags: -C panic=abort -Copt-level=0
//@ [WINDOWS_] compile-flags: --target=x86_64-pc-windows-msvc
//@ [WINDOWS_] needs-llvm-components: x86
//@ [ANDROID_] compile-flags: --target=armv7-linux-androideabi
//@ [ANDROID_] needs-llvm-components: arm

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK: attributes #{{.*}} uwtable
pub fn foo() {}
