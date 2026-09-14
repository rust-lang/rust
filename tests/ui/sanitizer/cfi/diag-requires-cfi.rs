// Verifies that `-Zsanitizer-cfi-diag` requires `-Zsanitizer=cfi`.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer-cfi-diag

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsanitizer-cfi-diag` requires `-Zsanitizer=cfi`
