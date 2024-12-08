#![crate_type = "bin"]
#![feature(lang_items, alloc_error_handler)]
#![no_main]
#![no_std]

use core::alloc::Layout;
use core::panic::PanicInfo;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

#[lang = "eh_personality"]
fn eh() {}

#[alloc_error_handler]
fn oom(_: Layout) -> ! {
    loop {}
}
