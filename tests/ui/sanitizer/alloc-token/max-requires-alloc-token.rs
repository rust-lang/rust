// Verifies that `-Zsanitizer-alloc-token-max` requires `-Zsanitizer=alloc-token`.
//
//@ needs-sanitizer-alloc-token
//@ compile-flags: -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer-alloc-token-max=4

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsanitizer-alloc-token-max` requires `-Zsanitizer=alloc-token`
