// Verifies that `-Zsanitizer=cfi` with `-Clto` or `-Clto=thin` requires `-Ccodegen-units=1`.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Ccodegen-units=2 -Clto -Ctarget-feature=-crt-static -Zsanitizer=cfi

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsanitizer=cfi` with `-Clto` requires `-Ccodegen-units=1`
