// For CFI, the helper flag -Zsanitizer-cfi-minimal-runtime should also be a target modifier.

//@ needs-sanitizer-cfi
//@ aux-build:cfi-minimal-runtime.rs

//@ revisions: ok wrong_flag wrong_sanitizer
//@[ok] compile-flags: -Clto -Zsanitizer=cfi -Zsanitizer-cfi-recover -Zsanitizer-cfi-minimal-runtime
//@[wrong_flag] compile-flags: -Clto -Zsanitizer=cfi -Zsanitizer-cfi-recover
//@[ok] check-pass

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]

extern crate cfi_minimal_runtime;

//[wrong_flag]~? ERROR mixing `-Zsanitizer-cfi-minimal-runtime` will cause an ABI mismatch in crate `sanitizer_cfi_minimal_runtime`
//[wrong_sanitizer]~? ERROR mixing `-Zsanitizer` will cause an ABI mismatch in crate `sanitizer_cfi_minimal_runtime`
