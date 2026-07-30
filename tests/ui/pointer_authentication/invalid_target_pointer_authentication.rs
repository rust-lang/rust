//@ ignore-backends: gcc
//@ check-pass
//@ needs-llvm-components: aarch64

//@ compile-flags: -Zunstable-options -Tpointer-authentication=-elf-got --crate-type=lib --target aarch64-unknown-linux-gnu

#![feature(no_core)]
#![no_std]
#![no_main]
#![no_core]
//~? WARN `-T pointer-authentication` is not supported for target aarch64-unknown-linux-gnu and will be ignored
