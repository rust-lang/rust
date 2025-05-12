// Verifies that `-Zsanitizer-cfi-normalize-integers` requires `-Zsanitizer=cfi` or
// `-Zsanitizer=kcfi`
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer-cfi-normalize-integers

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsanitizer-cfi-normalize-integers` requires `-Zsanitizer=cfi` or `-Zsanitizer=kcfi`
