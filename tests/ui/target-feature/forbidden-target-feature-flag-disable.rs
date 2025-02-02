//! Ensure "forbidden" target features cannot be disabled via `-Ctarget-feature`.
//@ compile-flags: --target=riscv32e-unknown-none-elf --crate-type=lib
//@ needs-llvm-components: riscv
//@ non-aux-compile-flags: -Ctarget-feature=-forced-atomics
//@ check-fail
//@ ignore-backends: gcc
//@ add-minicore

#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

//~? ERROR target feature `forced-atomics` cannot be disabled with `-Ctarget-feature`: unsound because it changes the ABI of atomic operations
