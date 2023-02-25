// aux-build:panic-handler.rs
// compile-flags: --document-private-items
// build-pass
// ignore-windows

#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
