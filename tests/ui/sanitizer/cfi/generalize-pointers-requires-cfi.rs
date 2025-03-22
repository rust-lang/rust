// Verifies that `-Zsanitizer-cfi-generalize-pointers` requires `-Csanitize=cfi` or
// `-Csanitize=kcfi`.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer-cfi-generalize-pointers

#![feature(no_core)]
#![no_core]
#![no_main]
