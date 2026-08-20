// Verifies that "alloc-token-fast-abi" module flag is added.
//
//@ needs-sanitizer-alloc-token
//@ min-llvm-version: 22
//@ compile-flags: -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer=alloc-token -Zsanitizer-alloc-token-fast-abi=yes

#![crate_type = "lib"]

pub fn foo() {}

// CHECK: !{{[0-9]+}} = !{i32 1, !"alloc-token-fast-abi", i32 1}
