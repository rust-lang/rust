//@ check-pass

#![no_std]
#![crate_type = "lib"]
#![allow(dead_code)]

fn cmp(a: *mut [u8], b: *mut [u8]) -> bool {
    a == b
    //~^ WARN ambiguous wide pointer comparison
}
