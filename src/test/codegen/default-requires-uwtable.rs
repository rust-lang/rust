// revisions: WINDOWS ANDROID
// needs-llvm-components: x86 arm
// compile-flags: -C panic=abort
// [WINDOWS] compile-flags: --target=x86_64-pc-windows-msvc
// [ANDROID] compile-flags: --target=armv7-linux-androideabi

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

#[lang = "sized"]
trait Sized {}

// CHECK: attributes #{{.*}} uwtable
pub fn foo() {}
