//@ ignore-backends: gcc
//@ check-pass
//@ needs-llvm-components: aarch64

//@ compile-flags: -Zpointer-authentication=-elf-got --crate-type=lib --target aarch64-unknown-linux-gnu

#![feature(no_core)]
#![no_std]
#![no_main]
#![no_core]
//~? WARN `-Z pointer-authentication` is not supported for target aarch64-unknown-linux-gnu and will be ignored
