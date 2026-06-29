// Verifies that `-Tsanitizer=cfi` with `-Clto` or `-Clto=thin` requires `-Ccodegen-units=1`.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Ccodegen-units=2 -Clto -Ctarget-feature=-crt-static -Tsanitizer=cfi -Zunstable-options

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Tsanitizer=cfi` with `-Clto` requires `-Ccodegen-units=1`
