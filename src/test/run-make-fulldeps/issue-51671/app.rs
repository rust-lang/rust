#![crate_type = "bin"]
#![feature(lang_items)]
#![feature(panic_implementation)]
#![no_main]
#![no_std]

use core::alloc::Layout;
use core::panic::PanicInfo;

#[panic_implementation]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

#[lang = "eh_personality"]
fn eh() {}

#[lang = "oom"]
fn oom(_: Layout) -> ! {
    loop {}
}
