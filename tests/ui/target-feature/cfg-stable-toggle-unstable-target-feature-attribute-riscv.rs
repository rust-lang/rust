//! Ensure cfg-only stable target_features trigger errors when enabled via attribute.
//@ compile-flags: --crate-type=lib
//@ compile-flags: --target=riscv64gc-unknown-none-elf
//@ needs-llvm-components: riscv
//@ add-minicore
//@ ignore-backends: gcc
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

#[target_feature(enable = "v")]
//~^ERROR: the target feature `v` is currently unstable
#[target_feature(enable = "f")]
//~^ERROR: the target feature `f` is allowed in cfg but unstable otherwise
pub unsafe fn my_fun() {}
