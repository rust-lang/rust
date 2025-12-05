//! Ensure ABI-incompatible features cannot be enabled via `#[target_feature]`.
// ignore-tidy-linelength
//@ compile-flags: --crate-type=lib
//@ revisions: x86 riscv
//@[x86] compile-flags: --target=x86_64-unknown-linux-gnu
//@[x86] needs-llvm-components: x86
//@[riscv] compile-flags: --target=riscv32e-unknown-none-elf
//@[riscv] needs-llvm-components: riscv
//@ ignore-backends: gcc
//@ add-minicore
#![feature(no_core, riscv_target_feature, x87_target_feature)]
#![no_core]

extern crate minicore;
use minicore::*;

#[cfg_attr(x86, target_feature(enable = "soft-float"))] #[cfg_attr(riscv, target_feature(enable = "d"))]
//~^ERROR: cannot be enabled with
pub unsafe fn my_fun() {}
