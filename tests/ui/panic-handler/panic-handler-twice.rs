//@ dont-check-compiler-stderr
//@ aux-build:some-panic-impl.rs
//~? ERROR multiple implementations of `#[panic_handler]`

#![feature(lang_items)]
#![no_std]
#![no_main]

extern crate some_panic_impl;

use core::panic::PanicInfo;
use core::panic_handler;

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    loop {}
}

#[lang = "eh_personality"]
fn eh() {}
