//@ no-prefer-dynamic

//@ needs-sanitizer-kcfi
//@ needs-sanitizer-safestack

//@ compile-flags: -C panic=abort -Zsanitizer=safestack,kcfi

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
