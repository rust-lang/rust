// Test that the `indirect_branch_cs_prefix` module attribute is (not)
// emitted when the `-Zindirect-branch-cs-prefix` flag is (not) set.

//@ add-minicore
//@ revisions: unset set
//@ needs-llvm-components: x86
//@ compile-flags: --target x86_64-unknown-linux-gnu
//@ [set] compile-flags: -Tindirect-branch-cs-prefix -Zunstable-options

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

// unset-NOT: !{{[0-9]+}} = !{i32 4, !"indirect_branch_cs_prefix", i32 1}
// set: !{{[0-9]+}} = !{i32 4, !"indirect_branch_cs_prefix", i32 1}
