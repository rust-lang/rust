// Verifies that `-Zsplit-lto-unit` requires `-Clto`, `-Clto=thin`, or `-Clinker-plugin-lto`.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsplit-lto-unit

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsplit-lto-unit` requires `-Clto`, `-Clto=thin`, or `-Clinker-plugin-lto`
