//@ no-prefer-dynamic
//@ needs-sanitizer-kcfi
//@ compile-flags: -Cpanic=abort -Zunstable-options -Csanitize=kcfi -Zsanitizer-cfi-normalize-integers

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
