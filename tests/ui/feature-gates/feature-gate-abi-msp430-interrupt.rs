//@ add-core-stubs
//@ needs-llvm-components: msp430
//@ compile-flags: --target=msp430-none-elf --crate-type=rlib
#![no_core]
#![feature(no_core, lang_items)]

extern crate minicore;
use minicore::*;

extern "msp430-interrupt" fn f() {}
//~^ ERROR "msp430-interrupt" ABI is experimental

trait T {
    extern "msp430-interrupt" fn m();
    //~^ ERROR "msp430-interrupt" ABI is experimental

    extern "msp430-interrupt" fn dm() {}
    //~^ ERROR "msp430-interrupt" ABI is experimental
}

struct S;
impl T for S {
    extern "msp430-interrupt" fn m() {}
    //~^ ERROR "msp430-interrupt" ABI is experimental
}

impl S {
    extern "msp430-interrupt" fn im() {}
    //~^ ERROR "msp430-interrupt" ABI is experimental
}

type TA = extern "msp430-interrupt" fn();
//~^ ERROR "msp430-interrupt" ABI is experimental

extern "msp430-interrupt" {}
//~^ ERROR "msp430-interrupt" ABI is experimental
