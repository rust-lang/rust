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
fn eh(
    _version: i32,
    _actions: i32,
    _exception_class: u64,
    _exception_object: *mut (),
    _context: *mut (),
) -> i32 {
    loop {}
}

#[alloc_error_handler]
fn oom(_: Layout) -> ! {
    loop {}
}
