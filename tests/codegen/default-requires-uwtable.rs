// revisions: WINDOWS ANDROID
// compile-flags: -C panic=abort -Copt-level=0
// [WINDOWS] compile-flags: --target=x86_64-pc-windows-msvc
// [WINDOWS] needs-llvm-components: x86
// [ANDROID] compile-flags: --target=armv7-linux-androideabi
// [ANDROID] needs-llvm-components: arm

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

#[lang = "sized"]
trait Sized {}

// CHECK: attributes #{{.*}} uwtable
pub fn foo() {}
