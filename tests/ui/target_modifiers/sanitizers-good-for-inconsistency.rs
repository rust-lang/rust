// AddressSanitizer, LeakSanitizer are good to be inconsistent (they are not a target modifiers)

//@ revisions: wrong_address_san wrong_leak_san

//@[wrong_address_san] needs-sanitizer-address
//@[wrong_leak_san] needs-sanitizer-leak

//@ aux-build:no-sanitizers.rs
//@ compile-flags: -Cpanic=abort -C target-feature=-crt-static

//@[wrong_address_san] compile-flags: -Zsanitizer=address
//@[wrong_leak_san] compile-flags: -Zsanitizer=leak
//@ check-pass

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]

extern crate no_sanitizers;
