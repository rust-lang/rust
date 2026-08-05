// Currently still just a warning.
//@ build-pass
//@ compile-flags: --crate-type=rlib --target=armv7-unknown-linux-gnueabihf
//@ needs-llvm-components: arm
//@ aux-build: using-atomics-32.rs
//@ ignore-backends: gcc
#![feature(no_core)]
#![no_core]

extern crate using_atomics_32;

//~? WARN mixing target features will cause an ABI mismatch
