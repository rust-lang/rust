// Verifies that "kcfi" module flag is added.
//
//@ add-core-stubs
//@ revisions: aarch64 x86_64
//@ [aarch64] compile-flags: --target aarch64-unknown-none
//@ [aarch64] needs-llvm-components: aarch64
//@ [x86_64] compile-flags: --target x86_64-unknown-none
//@ [x86_64] needs-llvm-components: x86
//@ compile-flags: -Ctarget-feature=-crt-static -Zsanitizer=kcfi

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

pub fn foo() {}

// CHECK: !{{[0-9]+}} = !{i32 4, !"kcfi", i32 1}
