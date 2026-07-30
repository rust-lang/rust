// Verifies that `-Zsanitizer=cfi` requires `-Clto` or `-Clinker-plugin-lto`.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Tsanitizer=cfi -Zunstable-options

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Tsanitizer=cfi` requires `-Clto` or `-Clinker-plugin-lto`
