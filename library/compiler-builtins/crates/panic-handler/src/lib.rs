//! This is needed for tests on targets that require a `#[panic_handler]` function

#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}
