// Verifies that `-Zsanitizer-cfi-normalize-integers` requires `-Csanitize=cfi` or
// `-Csanitize=kcfi`
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer-cfi-normalize-integers

#![feature(no_core)]
#![no_core]
#![no_main]
