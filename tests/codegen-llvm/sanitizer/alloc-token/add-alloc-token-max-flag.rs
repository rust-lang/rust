// Verifies that "alloc-token-max" module flag is added.
//
// The pointer-split heap partitioning scheme sets the maximum number of tokens to two (i.e.,
// `-Zsanitizer-alloc-token-max` is ignored), so that the token identifier is the partition number:
// token identifier 0 for the partition not containing pointers, and token identifier 1 for the
// partition containing pointers.
//
// The type-hash-pointer-split heap partitioning scheme uses a configurable maximum number of tokens
// (i.e., `-Zsanitizer-alloc-token-max`, defaulting to the same value as Clang, i.e., the number of
// tokens bounded by `SIZE_MAX`, when not set), so that the token identifier is derived from a
// stable hash of the allocated type name within each partition, providing per-type token
// identifiers.
//
//@ needs-sanitizer-alloc-token
//@ min-llvm-version: 22
//@ revisions: Default MaxIgnored Scheme SchemeMax
//@ compile-flags: -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer=alloc-token
//@ [MaxIgnored] compile-flags: -Zsanitizer-alloc-token-max=4
//@ [Scheme] compile-flags: -Zsanitizer-alloc-token-scheme=type-hash-pointer-split
//@ [SchemeMax] compile-flags: -Zsanitizer-alloc-token-scheme=type-hash-pointer-split
//@ [SchemeMax] compile-flags: -Zsanitizer-alloc-token-max=4

#![crate_type = "lib"]

pub fn foo() {}

// Default: !{{[0-9]+}} = !{i32 1, !"alloc-token-max", i32 2}
// MaxIgnored: !{{[0-9]+}} = !{i32 1, !"alloc-token-max", i32 2}
// Scheme-NOT: !"alloc-token-max"
// SchemeMax: !{{[0-9]+}} = !{i32 1, !"alloc-token-max", i32 4}
