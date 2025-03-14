//@ add-core-stubs
//@ build-fail
//@ compile-flags: --target i686-unknown-linux-gnu --crate-type lib
//@ needs-llvm-components: x86
#![feature(no_core, lang_items)]
#![allow(internal_features)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

// 0x7fffffff is fine, but after rounding up it becomes too big
#[repr(C, align(2))]
pub struct Example([u8; 0x7fffffff]);

pub fn lib(_x: Example) {} //~ERROR: too big for the target architecture
