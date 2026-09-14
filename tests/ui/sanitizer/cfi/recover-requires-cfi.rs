// Verifies that `-Zsanitizer-cfi-recover` requires `-Zsanitizer=cfi`.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer-cfi-recover

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsanitizer-cfi-recover` requires `-Zsanitizer=cfi`
