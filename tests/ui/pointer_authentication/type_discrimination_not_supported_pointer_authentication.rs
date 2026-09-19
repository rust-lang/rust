//@ ignore-backends: gcc
//@ check-fail
//@ needs-llvm-components: aarch64

//@ compile-flags: -Zpointer-authentication=+function-pointer-type-discrimination --crate-type=lib --target aarch64-unknown-linux-pauthtest

#![feature(no_core)]
#![no_std]
#![no_main]
#![no_core]

//~? ERROR function pointer type discrimination is not supported
