#![no_std]

#[panic_handler]
fn handler(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

pub unsafe fn oops(x: *const u32) -> u32 {
    *x
}
