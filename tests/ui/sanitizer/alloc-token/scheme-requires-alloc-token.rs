// Verifies that `-Zsanitizer-alloc-token-scheme` requires `-Zsanitizer=alloc-token`.
//
//@ needs-sanitizer-alloc-token
//@ compile-flags: -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer-alloc-token-scheme=pointer-split

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsanitizer-alloc-token-scheme` requires `-Zsanitizer=alloc-token`
