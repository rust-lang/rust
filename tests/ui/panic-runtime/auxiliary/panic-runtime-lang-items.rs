//@ no-prefer-dynamic

#![crate_type = "rlib"]

#![no_std]
#![feature(lang_items)]

use core::panic::PanicInfo;

#[panic_handler]
fn panic_impl(info: &PanicInfo) -> ! { loop {} }
#[lang = "eh_personality"]
fn eh_personality() {}
