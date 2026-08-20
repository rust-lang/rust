// Verifies that `-Zsanitizer-alloc-token-fast-abi` requires `-Zsanitizer=alloc-token`.
//
//@ needs-sanitizer-alloc-token
//@ compile-flags: -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer-alloc-token-fast-abi=yes

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsanitizer-alloc-token-fast-abi` requires `-Zsanitizer=alloc-token`
