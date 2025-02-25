//@ add-core-stubs
//@ needs-llvm-components: avr
//@ compile-flags: --target=avr-none -C target-cpu=atmega328p --crate-type=rlib
#![no_core]
#![feature(no_core, lang_items)]

extern crate minicore;
use minicore::*;

// Test that the AVR interrupt ABI cannot be used when avr_interrupt
// feature gate is not used.

extern "avr-non-blocking-interrupt" fn fu() {}
//~^ ERROR extern "avr-non-blocking-interrupt" ABI is experimental
extern "avr-interrupt" fn f() {}
//~^ ERROR extern "avr-interrupt" ABI is experimental

trait T {
    extern "avr-interrupt" fn m();
    //~^ ERROR extern "avr-interrupt" ABI is experimental
    extern "avr-non-blocking-interrupt" fn mu();
    //~^ ERROR extern "avr-non-blocking-interrupt" ABI is experimental

    extern "avr-interrupt" fn dm() {}
    //~^ ERROR extern "avr-interrupt" ABI is experimental
    extern "avr-non-blocking-interrupt" fn dmu() {}
    //~^ ERROR extern "avr-non-blocking-interrupt" ABI is experimental
}

struct S;
impl T for S {
    extern "avr-interrupt" fn m() {}
    //~^ ERROR extern "avr-interrupt" ABI is experimental
    extern "avr-non-blocking-interrupt" fn mu() {}
    //~^ ERROR extern "avr-non-blocking-interrupt" ABI is experimental
}

impl S {
    extern "avr-interrupt" fn im() {}
    //~^ ERROR extern "avr-interrupt" ABI is experimental
    extern "avr-non-blocking-interrupt" fn imu() {}
    //~^ ERROR extern "avr-non-blocking-interrupt" ABI is experimental
}

type TA = extern "avr-interrupt" fn();
//~^ ERROR extern "avr-interrupt" ABI is experimental
type TAU = extern "avr-non-blocking-interrupt" fn();
//~^ ERROR extern "avr-non-blocking-interrupt" ABI is experimental

extern "avr-interrupt" {}
//~^ ERROR extern "avr-interrupt" ABI is experimental
extern "avr-non-blocking-interrupt" {}
//~^ ERROR extern "avr-non-blocking-interrupt" ABI is experimental
