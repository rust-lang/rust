// Test that `download-rustc` doesn't put duplicate copies of libc in the sysroot.
//@ check-pass
#![crate_type = "lib"]
#![no_std]
#![feature(rustc_private)]

extern crate libc;
