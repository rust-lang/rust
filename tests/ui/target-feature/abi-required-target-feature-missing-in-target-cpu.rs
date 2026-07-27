//! Sometimes `-Ctarget-cpu` can *disable* target features that would by default be enabled on the
//! current target. Ensure that we catch the case where those target features are important for the
//! ABI.

//@ compile-flags: --crate-type=lib
//@ revisions: x86 arm
//@[x86] compile-flags: --target=i686-unknown-linux-gnu -Ctarget-cpu=pentium
//@[x86] needs-llvm-components: x86
//@[arm] compile-flags: --target=armv8r-none-eabihf -Ctarget-cpu=cortex-r4
//@[arm] needs-llvm-components: arm

// For now this is just a warning.
//@ build-pass
//@ ignore-backends: gcc
//@ add-minicore

#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

//~? WARN must be enabled to ensure that the ABI of the current target can be implemented correctly
