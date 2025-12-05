//! Ensure ABI-incompatible features cannot be enabled via `-Ctarget-feature`.
// These are just warnings for now.
//@ check-pass
//@ compile-flags: --crate-type=lib
//@ revisions: x86 riscv
//@[x86] compile-flags: --target=x86_64-unknown-linux-gnu -Ctarget-feature=+soft-float
//@[x86] needs-llvm-components: x86
//@[riscv] compile-flags: --target=riscv32e-unknown-none-elf -Ctarget-feature=+d
// FIXME(#147881): *disable* the feature again for minicore as otherwise that will fail to build.
//@[riscv] minicore-compile-flags: -Ctarget-feature=-d
//@[riscv] needs-llvm-components: riscv
//@ ignore-backends: gcc
//@ add-minicore

#![feature(no_core, riscv_target_feature)]
#![no_core]

extern crate minicore;
use minicore::*;

//~? WARN must be disabled to ensure that the ABI of the current target can be implemented correctly
//~? WARN unstable feature specified for `-Ctarget-feature`
