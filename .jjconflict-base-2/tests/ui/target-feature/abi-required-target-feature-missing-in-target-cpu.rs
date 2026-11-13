//! Sometimes `-Ctarget-cpu` can *disable* target features that would by default be enabled on the
//! current target. Ensure that we catch the case where those target features are important for the
//! ABI.

//@ compile-flags: --crate-type=lib
//@ revisions: x86 arm
//@[x86] compile-flags: --target=i686-unknown-linux-gnu -Ctarget-cpu=pentium
//@[x86] needs-llvm-components: x86
//@[arm] compile-flags: --target=armv8r-none-eabihf -Ctarget-cpu=cortex-r4
//@[arm] needs-llvm-components: arm

// check-fail

//@ ignore-backends: gcc
//@ add-minicore
// Don't inherit the target-cpu above for minicore, to avoid errors when building that.
//@ minicore-compile-flags: -Ctarget-cpu=generic
//@[x86] minicore-compile-flags: -Ctarget-feature=+sse2

#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

//[x86,arm]~? ERROR must be enabled to ensure that the ABI of the current target can be implemented correctly
