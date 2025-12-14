// Verifies that `-Csanitize=cfi` requires `-Clto` or `-Clinker-plugin-lto`.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitize -Zunstable-options -Csanitize=cfi

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Csanitize=cfi` requires `-Clto` or `-Clinker-plugin-lto`
