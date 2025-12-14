//@ known-bug: #126418
//@ compile-flags: --crate-type=lib
//@ only-x86_64
//@ ignore-backends: gcc
#![feature(abi_x86_interrupt)]
pub extern "x86-interrupt" fn f(_: ()) {}
