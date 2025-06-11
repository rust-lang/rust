//@ add-core-stubs
//@ compile-flags: --crate-type=lib --target x86_64-unknown-none
//@ needs-llvm-components: x86
//@ edition: 2018
#![no_core]
#![feature(no_core, lang_items)]
extern crate minicore;
use minicore::*;

// Check we error before unsupported ABIs reach codegen stages.

fn anything() {
    let a = unsafe { mem::transmute::<usize, extern "thiscall" fn(i32)>(4) }(2);
    //~^ ERROR: is not a supported ABI for the current target [E0570]
}
