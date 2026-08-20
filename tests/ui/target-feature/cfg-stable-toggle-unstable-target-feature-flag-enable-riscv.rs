//! Ensure cfg-only stable target_features trigger warnings when enabled via compile flag.
//@ check-pass
//@ compile-flags: --crate-type=lib
//@ compile-flags: --target=riscv64gc-unknown-none-elf -Ctarget-feature=+v -Ctarget-feature=+f
// FIXME(#147881): *disable* the feature again for minicore as otherwise that will fail to build.
//@ minicore-compile-flags: -Ctarget-feature=-v -Ctarget-feature=-f
//@ needs-llvm-components: riscv
//@ ignore-backends: gcc
//@ add-minicore
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

//~? WARN unstable feature specified for `-Ctarget-feature`
//~? WARN unstable feature specified for `-Ctarget-feature`
