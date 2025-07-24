// Verifies that "kcfi-arity" module flag is added.
//
//@ add-core-stubs
//@ revisions: x86_64
//@ [x86_64] compile-flags: --target x86_64-unknown-none
//@ [x86_64] needs-llvm-components: x86
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Zsanitizer=kcfi -Zsanitizer-kcfi-arity
//@ min-llvm-version: 21.0.0

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

pub fn foo() {}

// CHECK: !{{[0-9]+}} = !{i32 4, !"kcfi-arity", i32 1}
