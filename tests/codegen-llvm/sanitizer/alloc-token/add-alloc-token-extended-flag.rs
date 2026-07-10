// Verifies that "alloc-token-extended" module flag is added.
//
//@ needs-sanitizer-alloc-token
//@ min-llvm-version: 22
//@ revisions: Default NoExtended
//@ compile-flags: -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer=alloc-token
//@ [NoExtended] compile-flags: -Zsanitizer-alloc-token-extended=no

#![crate_type = "lib"]

pub fn foo() {}

// Default: !{{[0-9]+}} = !{i32 1, !"alloc-token-extended", i32 1}
// NoExtended-NOT: !"alloc-token-extended"
