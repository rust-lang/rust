// Verifies that `-Csanitize=cfi` with `-Clto` or `-Clto=thin` requires `-Ccodegen-units=1`.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Ccodegen-units=2 -Clto -Ctarget-feature=-crt-static -Zunstable-options -Csanitize=cfi

#![feature(no_core)]
#![no_core]
#![no_main]
