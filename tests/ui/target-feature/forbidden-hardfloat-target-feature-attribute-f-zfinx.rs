//! Ensure ABI-incompatible features cannot be enabled via `#[target_feature]`.
//@ compile-flags: --target=riscv64gc-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: riscv
//@ ignore-backends: gcc
//@ add-minicore
#![feature(no_core, riscv_target_feature)]
#![no_core]

extern crate minicore;
use minicore::*;

#[target_feature(enable = "zdinx")]
//~^ERROR: cannot be enabled with
pub unsafe fn my_fun() {}
