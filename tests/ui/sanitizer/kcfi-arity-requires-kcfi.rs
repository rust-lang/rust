// Verifies that `-Zsanitizer-kcfi-arity` requires `-Zsanitizer=kcfi`.
//
//@ needs-sanitizer-kcfi
//@ compile-flags: -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer-kcfi-arity

#![feature(no_core)]
#![no_core]
#![no_main]
