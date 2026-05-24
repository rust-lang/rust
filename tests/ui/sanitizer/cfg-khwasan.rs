// Verifies that when compiling with -Csanitize=kernel-hwaddress,
// the `#[cfg(sanitize = "hwaddress")]` attribute is configured.

//@ add-minicore
//@ check-pass
//@ compile-flags: -Cunsafe-allow-abi-mismatch=sanitize -Zunstable-options -Csanitize=kernel-hwaddress --target aarch64-unknown-none
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
