// Verifies that `-Tsanitizer-cfi-normalize-integers` requires `-Tsanitizer=cfi` or
// `-Tsanitizer=kcfi`
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static
//@ compile-flags: -Tsanitizer-cfi-normalize-integers -Zunstable-options

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Tsanitizer-cfi-normalize-integers` requires `-Tsanitizer=cfi` or `-Tsanitizer=kcfi`
