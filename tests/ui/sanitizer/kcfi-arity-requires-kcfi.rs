// Verifies that `-Zsanitizer-kcfi-arity` requires `-Csanitize=kcfi`.
//
//@ needs-sanitizer-kcfi
//@ compile-flags: -Ctarget-feature=-crt-static -Zsanitizer-kcfi-arity

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsanitizer-kcfi-arity` requires `-Csanitize=kcfi`
