//@ dont-check-compiler-stderr
//@ aux-build:some-panic-impl.rs

#![feature(lang_items)]
#![feature(panic_unwind)]
#![no_std]
#![no_main]

extern crate some_panic_impl;
extern crate unwind;

use core::panic::PanicInfo;

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    //~^ ERROR found duplicate lang item `panic_impl`
    loop {}
}
