#![crate_type = "bin"]
#![feature(lang_items, alloc_error_handler, panic_unwind)]
#![no_main]
#![no_std]

extern crate unwind;

use core::alloc::Layout;
use core::panic::PanicInfo;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

#[alloc_error_handler]
fn oom(_: Layout) -> ! {
    loop {}
}
