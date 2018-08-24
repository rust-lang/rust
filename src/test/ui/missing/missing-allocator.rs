// compile-flags: -C panic=abort
// no-prefer-dynamic

#![no_std]
#![crate_type = "staticlib"]
#![feature(panic_implementation, alloc_error_handler, alloc)]

#[panic_implementation]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[alloc_error_handler]
fn oom(_: core::alloc::Layout) -> ! {
    loop {}
}

extern crate alloc;
