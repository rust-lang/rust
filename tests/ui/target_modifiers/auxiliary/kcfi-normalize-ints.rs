//@ no-prefer-dynamic
//@ needs-sanitizer-kcfi
//@ compile-flags: -C panic=abort -Zsanitizer=kcfi -Zsanitizer-cfi-normalize-integers

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
