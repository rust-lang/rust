// compile-flags: -C panic=abort
// no-prefer-dynamic

#![no_std]
#![crate_type = "staticlib"]
#![feature(panic_handler, alloc_error_handler)]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[alloc_error_handler]
fn oom(_: core::alloc::Layout) -> ! {
    loop {}
}

extern crate alloc;
