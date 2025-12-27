// Verifies that `-Zsanitizer-cfi-generalize-pointers` requires `-Csanitize=cfi` or
// `-Csanitize=kcfi`.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Ccodegen-units=1 -Clto -Ctarget-feature=-crt-static -Zsanitizer-cfi-generalize-pointers

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsanitizer-cfi-generalize-pointers` requires `-Csanitize=cfi` or `-Csanitize=kcfi`
