// For kCFI, the helper flag -Zsanitizer-cfi-normalize-integers should also be a target modifier.

//@ needs-sanitizer-kcfi
//@ aux-build:kcfi-normalize-ints.rs
//@ compile-flags: -Cpanic=abort

//@ revisions: ok wrong_flag wrong_sanitizer
//@[ok] compile-flags: -Zsanitizer=kcfi -Zsanitizer-cfi-normalize-integers
//@[wrong_flag] compile-flags: -Zsanitizer=kcfi
//@[ok] check-pass

#![feature(no_core)]
//[wrong_flag]~^ ERROR mixing `-Zsanitizer-cfi-normalize-integers` will cause an ABI mismatch in crate `sanitizer_kcfi_normalize_ints`
//[wrong_sanitizer]~^^ ERROR mixing `-Zsanitizer` will cause an ABI mismatch in crate `sanitizer_kcfi_normalize_ints`
#![crate_type = "rlib"]
#![no_core]

extern crate kcfi_normalize_ints;
