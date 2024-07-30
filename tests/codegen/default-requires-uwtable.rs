//@ revisions: windows android
//@ compile-flags: -C panic=abort -Copt-level=0
//@ [windows] compile-flags: --target=x86_64-pc-windows-msvc
//@ [windows] needs-llvm-components: x86
//@ [android] compile-flags: --target=armv7-linux-androideabi
//@ [android] needs-llvm-components: arm

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

#[lang = "sized"]
trait Sized {}

// CHECK: attributes #{{.*}} uwtable
pub fn foo() {}
