// Verifies that unstable supported sanitizers can be used with `-Zunstable-options`.
//
//@ needs-llvm-components: x86
//@ compile-flags: -Zunstable-options -Csanitize=kernel-address --target x86_64-unknown-none
//@ build-pass

#![crate_type = "rlib"]
#![feature(no_core)]
#![no_core]
#![no_main]
