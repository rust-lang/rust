// Verifies that `-Csanitizer=cfi` requires `-Clto` or `-Clinker-plugin-lto`.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zunstable-options -Csanitizer=cfi

#![feature(no_core)]
#![no_core]
#![no_main]
