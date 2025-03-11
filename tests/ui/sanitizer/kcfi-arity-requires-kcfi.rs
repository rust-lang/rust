// Verifies that `-Zsanitizer-kcfi-arity` requires `-Zsanitizer=kcfi`.
//
//@ needs-sanitizer-kcfi
//@ compile-flags: -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer-kcfi-arity
//@ min-llvm-version: 20.1.0

#![feature(no_core)]
#![no_core]
#![no_main]
