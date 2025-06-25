//@ add-core-stubs
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
#![feature(abi_cmse_nonsecure_call, lang_items, no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

extern "cmse-nonsecure-call" { //~ ERROR [E0781]
    fn test();
}
