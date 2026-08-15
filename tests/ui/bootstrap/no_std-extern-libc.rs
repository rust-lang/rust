// Test that `download-rustc` doesn't put duplicate copies of libc in the sysroot.
//@ check-pass
//@ ignore-windows doesn't necessarily have the libc crate
#![crate_type = "lib"]
#![no_std]
#![feature(rustc_private)]

extern crate libc;
