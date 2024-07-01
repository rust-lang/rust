// Verifies that `-Zsanitizer-cfi-generalize-pointers` requires `-Csanitizer=cfi` or
// `-Csanitizer=kcfi`.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer-cfi-generalize-pointers

#![feature(no_core)]
#![no_core]
#![no_main]
