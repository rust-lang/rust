// Verifies that `-Zsanitizer-kcfi-arity` requires LLVM 21.0.0 or later.
//
//@ needs-sanitizer-kcfi
//@ compile-flags: -Cpanic=abort -Ctarget-feature=-crt-static -Zunstable-options -Csanitize=kcfi -Zsanitizer-kcfi-arity
//@ build-fail
//@ max-llvm-major-version: 20

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsanitizer-kcfi-arity` requires LLVM 21.0.0 or later.
