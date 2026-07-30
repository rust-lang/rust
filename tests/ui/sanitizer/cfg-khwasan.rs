// Verifies that when compiling with -Zsanitizer=kernel-hwaddress,
// the `#[cfg(sanitize = "hwaddress")]` attribute is configured.

//@ add-minicore
//@ check-pass
//@ compile-flags: -Tsanitizer=kernel-hwaddress --target aarch64-unknown-none -Zunstable-options
//@ needs-llvm-components: aarch64
//@ ignore-backends: gcc

#![crate_type = "rlib"]
#![feature(cfg_sanitize, no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

const _: fn() -> () = main;

#[cfg(sanitize = "hwaddress")]
fn main() {}
