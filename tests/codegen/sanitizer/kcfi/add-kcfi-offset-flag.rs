// Verifies that "kcfi-offset" module flag is added.
//
//@ revisions: aarch64 x86_64
//@ [aarch64] compile-flags: --target aarch64-unknown-none
//@ [aarch64] needs-llvm-components: aarch64
//@ [x86_64] compile-flags: --target x86_64-unknown-none
//@ [x86_64] needs-llvm-components: x86
//@ compile-flags: -Ctarget-feature=-crt-static -Zsanitizer=kcfi -Z patchable-function-entry=4,3

#![feature(no_core, lang_items, patchable_function_entry)]
#![crate_type = "lib"]
#![no_core]

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

pub fn foo() {}

// CHECK: !{{[0-9]+}} = !{i32 4, !"kcfi-offset", i32 3}
