// Verifies that `-Zsanitizer=cfi` requires `-Clto` or `-Clinker-plugin-lto`.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsanitizer=cfi` requires `-Clto` or `-Clinker-plugin-lto`
