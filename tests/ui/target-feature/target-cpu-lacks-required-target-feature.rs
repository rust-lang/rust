//@ compile-flags: --target=i686-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: x86
//@ compile-flags: -Ctarget-cpu=pentium
// For now this is just a warning.
//@ build-pass
//@ ignore-backends: gcc
//@ add-minicore

#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

//~? WARN target feature `sse2` must be enabled to ensure that the ABI of the current target can be implemented correctly
