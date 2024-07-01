// Verifies that `-Zsanitizer-cfi-normalize-integers` requires `-Csanitizer=cfi` or
// `-Csanitizer=kcfi`
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer-cfi-normalize-integers

#![feature(no_core)]
#![no_core]
#![no_main]
