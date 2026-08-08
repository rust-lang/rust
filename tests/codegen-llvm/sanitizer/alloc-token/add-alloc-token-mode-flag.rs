// Verifies that "alloc-token-mode" module flag is added.
//
//@ needs-sanitizer-alloc-token
//@ min-llvm-version: 22
//@ compile-flags: -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer=alloc-token

#![crate_type = "lib"]

pub fn foo() {}

// CHECK: !{{[0-9]+}} = !{i32 1, !"alloc-token-mode", !"typehashpointersplit"}
