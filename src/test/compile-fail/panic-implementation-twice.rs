// aux-build:some-panic-impl.rs

#![feature(panic_implementation)]
#![feature(lang_items)]
#![no_std]
#![no_main]

extern crate some_panic_impl;

use core::panic::PanicInfo;

#[panic_implementation]
fn panic(info: &PanicInfo) -> ! {
    //~^ error duplicate lang item found: `panic_impl`
    loop {}
}

#[lang = "eh_personality"]
fn eh() {}
