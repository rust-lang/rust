//@ compile-flags:-C panic=abort

#![feature(lang_items)]
#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    //~^ ERROR multiple implementations of `#[panic_handler]`
    loop {}
}

#[panic_handler]
fn panic2(info: &PanicInfo) -> ! {
    loop {}
}
