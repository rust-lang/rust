//@ add-minicore
//@ needs-llvm-components: xtensa
//@ min-llvm-version: 22
//@ compile-flags: --target=xtensa-esp32-none-elf --crate-type=rlib
//@ ignore-backends: gcc
#![no_core]
#![feature(no_core, lang_items)]

extern crate minicore;
use minicore::*;

#[target_feature(enable = "bool")]
//~^ ERROR: currently unstable
unsafe fn foo() {}
