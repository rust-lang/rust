//! Ensure "forbidden" target features cannot be enabled via `#[target_feature]`.
//@ compile-flags: --target=riscv32e-unknown-none-elf --crate-type=lib
//@ needs-llvm-components: riscv
//@ ignore-backends: gcc
//@ add-minicore
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

#[target_feature(enable = "forced-atomics")]
//~^ERROR: cannot be enabled with
pub unsafe fn my_fun() {}
