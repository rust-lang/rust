//@ no-prefer-dynamic

//@ needs-sanitizer-kcfi
//@ needs-sanitizer-safestack

//@ compile-flags: -Cpanic=abort -Zunstable-options -Csanitize=safestack,kcfi

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
