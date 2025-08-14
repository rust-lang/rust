//@ add-core-stubs
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
#![feature(abi_c_cmse_nonsecure_call, lang_items, no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

pub extern "C-cmse-nonsecure-call" fn test() {} //~ ERROR [E0781]
