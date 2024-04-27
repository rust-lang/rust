//@ no-prefer-dynamic

#![crate_type = "rlib"]

#![no_std]
#![feature(lang_items)]

use core::panic::PanicInfo;

#[lang = "panic_impl"]
fn panic_impl(info: &PanicInfo) -> ! { loop {} }
#[lang = "eh_personality"]
fn eh_personality() {}
#[lang = "eh_catch_typeinfo"]
static EH_CATCH_TYPEINFO: u8 = 0;
