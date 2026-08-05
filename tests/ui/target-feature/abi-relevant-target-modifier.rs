// Currently still just a warning.
//@ build-pass
//@ revisions: baseline allowed
//@ compile-flags: --crate-type=rlib --target=armv7-unknown-linux-gnueabihf
//@ needs-llvm-components: arm
//@[allowed] compile-flags: -Cunsafe-allow-abi-mismatch=target-feature
//@ aux-build: using-atomics-32.rs
//@ ignore-backends: gcc
#![feature(no_core)]
#![no_core]

extern crate using_atomics_32;

//[baseline]~? WARN mixing target features will cause an ABI mismatch
