// Verifies that "kcfi-offset" module flag is added.
//
//@ add-minicore
//@ revisions: aarch64 x86_64
//@ [aarch64] compile-flags: --target aarch64-unknown-none
//@ [aarch64] needs-llvm-components: aarch64
//@ [x86_64] compile-flags: --target x86_64-unknown-none
//@ [x86_64] needs-llvm-components: x86
//@ compile-flags: -Copt-level=0 -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitize -Zunstable-options -Csanitize=kcfi -Zpatchable-function-entry=4,3

#![feature(no_core, lang_items, patchable_function_entry)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

pub fn foo() {}

// CHECK: !{{[0-9]+}} = !{i32 4, !"kcfi-offset", i32 3}
