// Verifies that `-Zsanitizer-alloc-token-fast-abi` requires a maximum number of tokens.
//
//@ needs-sanitizer-alloc-token
//@ compile-flags: -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=alloc-token -Zsanitizer-alloc-token-fast-abi=yes -Zsanitizer-alloc-token-scheme=type-hash-pointer-split

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsanitizer-alloc-token-fast-abi` requires a maximum number of tokens (i.e., `-Zsanitizer-alloc-token-max`)
