// Verifies that `-Zsanitizer-kcfi-arity` requires LLVM 21.0.0 or later.
//
//@ needs-sanitizer-kcfi
//@ compile-flags: -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Cpanic=abort -Zsanitizer=kcfi -Zsanitizer-kcfi-arity
//@ build-fail
//@ max-llvm-major-version: 20

//~? ERROR `-Zsanitizer-kcfi-arity` requires LLVM 21.0.0 or later.
#![feature(no_core)]
#![no_core]
#![no_main]
