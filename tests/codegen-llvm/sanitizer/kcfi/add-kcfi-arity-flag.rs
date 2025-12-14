// Verifies that "kcfi-arity" module flag is added.
//
//@ add-minicore
//@ revisions: x86_64
//@ [x86_64] compile-flags: --target x86_64-unknown-none
//@ [x86_64] needs-llvm-components: x86
//@ compile-flags: -Copt-level=0 -Ctarget-feature=-crt-static -Cpanic=abort -Zunstable-options -Csanitize=kcfi -Zsanitizer-kcfi-arity -Cunsafe-allow-abi-mismatch=sanitize
//@ min-llvm-version: 21.0.0

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

pub fn foo() {}

// CHECK: !{{[0-9]+}} = !{i32 4, !"kcfi-arity", i32 1}
