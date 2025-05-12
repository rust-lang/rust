// Regression test for #54505 - range borrowing suggestion had
// incorrect syntax (missing parentheses).

// This test doesn't use std
// (so all Ranges resolve to core::ops::Range...)

#![no_std]
#![feature(lang_items)]

use core::ops::RangeBounds;

#[cfg(any(not(target_arch = "wasm32"), target_os = "emscripten"))]
#[lang = "eh_personality"]
extern "C" fn eh_personality() {}
#[cfg(target_os = "emscripten")]
#[lang = "eh_catch_typeinfo"]
static EH_CATCH_TYPEINFO: u8 = 0;

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    unimplemented!();
}

// take a reference to any built-in range
fn take_range(_r: &impl RangeBounds<i8>) {}


fn main() {
    take_range(0..1);
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &(

    take_range(1..);
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &(

    take_range(..);
    //~^ ERROR mismatched types [E0308]
    //~| HELP you might have meant
    //~| HELP consider borrowing here
    //~| SUGGESTION &(

    take_range(0..=1);
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &(

    take_range(..5);
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &(

    take_range(..=42);
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &(
}
