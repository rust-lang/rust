//@ no-prefer-dynamic

//@ needs-sanitizer-kcfi
//@ needs-sanitizer-safestack

//@ compile-flags: -C panic=abort -Tsanitizer=safestack,kcfi -Zunstable-options

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
