// Verifies that `-Zsanitizer-cfi-minimal-runtime` requires
// `-Zsanitizer-cfi-recover` or `-Zsanitizer-cfi-diag`.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer-cfi-minimal-runtime

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsanitizer-cfi-minimal-runtime` requires `-Zsanitizer-cfi-recover` or `-Zsanitizer-cfi-diag`
