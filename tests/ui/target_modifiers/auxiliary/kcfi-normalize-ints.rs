//@ no-prefer-dynamic
//@ needs-sanitizer-kcfi
//@ compile-flags: -C panic=abort -Tsanitizer=kcfi -Tsanitizer-cfi-normalize-integers -Zunstable-options

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
