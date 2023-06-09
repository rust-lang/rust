// needs-llvm-components: avr
// compile-flags: --target=avr-unknown-gnu-atmega328 --crate-type=rlib
#![no_core]
#![feature(no_core, lang_items)]
#[lang="sized"]
trait Sized { }

// Test that the AVR interrupt ABI cannot be used when avr_interrupt
// feature gate is not used.

extern "avr-non-blocking-interrupt" fn fu() {}
//~^ ERROR avr-interrupt and avr-non-blocking-interrupt ABIs are experimental
extern "avr-interrupt" fn f() {}
//~^ ERROR avr-interrupt and avr-non-blocking-interrupt ABIs are experimental

trait T {
    extern "avr-interrupt" fn m();
    //~^ ERROR avr-interrupt and avr-non-blocking-interrupt ABIs are experimental
    extern "avr-non-blocking-interrupt" fn mu();
    //~^ ERROR avr-interrupt and avr-non-blocking-interrupt ABIs are experimental

    extern "avr-interrupt" fn dm() {}
    //~^ ERROR avr-interrupt and avr-non-blocking-interrupt ABIs are experimental
    extern "avr-non-blocking-interrupt" fn dmu() {}
    //~^ ERROR avr-interrupt and avr-non-blocking-interrupt ABIs are experimental
}

struct S;
impl T for S {
    extern "avr-interrupt" fn m() {}
    //~^ ERROR avr-interrupt and avr-non-blocking-interrupt ABIs are experimental
    extern "avr-non-blocking-interrupt" fn mu() {}
    //~^ ERROR avr-interrupt and avr-non-blocking-interrupt ABIs are experimental
}

impl S {
    extern "avr-interrupt" fn im() {}
    //~^ ERROR avr-interrupt and avr-non-blocking-interrupt ABIs are experimental
    extern "avr-non-blocking-interrupt" fn imu() {}
    //~^ ERROR avr-interrupt and avr-non-blocking-interrupt ABIs are experimental
}

type TA = extern "avr-interrupt" fn();
//~^ ERROR avr-interrupt and avr-non-blocking-interrupt ABIs are experimental
type TAU = extern "avr-non-blocking-interrupt" fn();
//~^ ERROR avr-interrupt and avr-non-blocking-interrupt ABIs are experimental

extern "avr-interrupt" {}
//~^ ERROR avr-interrupt and avr-non-blocking-interrupt ABIs are experimental
extern "avr-non-blocking-interrupt" {}
//~^ ERROR avr-interrupt and avr-non-blocking-interrupt ABIs are experimental
