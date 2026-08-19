// For kCFI, the helper flag -Tsanitizer-cfi-normalize-integers should also be a target modifier.

//@ needs-sanitizer-kcfi
//@ aux-build:kcfi-normalize-ints.rs
//@ compile-flags: -Cpanic=abort

//@ revisions: ok wrong_flag wrong_sanitizer
//@[ok] compile-flags: -Tsanitizer=kcfi -Tsanitizer-cfi-normalize-integers -Zunstable-options
//@[wrong_flag] compile-flags: -Tsanitizer=kcfi -Zunstable-options
//@[ok] check-pass

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]

extern crate kcfi_normalize_ints;

//[wrong_flag]~? ERROR mixing `-Tsanitizer-cfi-normalize-integers` will cause an ABI mismatch in crate `sanitizer_kcfi_normalize_ints`
//[wrong_sanitizer]~? ERROR mixing `-Tsanitizer` will cause an ABI mismatch in crate `sanitizer_kcfi_normalize_ints`
//[wrong_sanitizer]~? ERROR mixing `-Tsanitizer-cfi-normalize-integers` will cause an ABI mismatch in crate `sanitizer_kcfi_normalize_ints`
