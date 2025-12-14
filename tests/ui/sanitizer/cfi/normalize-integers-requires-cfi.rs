// Verifies that `-Zsanitizer-cfi-normalize-integers` requires `-Csanitize=cfi` or
// `-Csanitize=kcfi`
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Ccodegen-units=1 -Clto -Ctarget-feature=-crt-static -Zsanitizer-cfi-normalize-integers

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsanitizer-cfi-normalize-integers` requires `-Csanitize=cfi` or `-Csanitize=kcfi`
