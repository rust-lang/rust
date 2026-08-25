// Verifies that `-Zsanitizer-cfi-canonical-jump-tables` requires `-Zsanitizer=cfi`.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer-cfi-canonical-jump-tables=false

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsanitizer-cfi-canonical-jump-tables` requires `-Zsanitizer=cfi`
