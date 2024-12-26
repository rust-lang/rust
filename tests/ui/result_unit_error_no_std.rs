#![feature(lang_items, start, libc)]
#![no_std]
#![warn(clippy::result_unit_err)]

#[clippy::msrv = "1.80"]
pub fn returns_unit_error_no_lint() -> Result<u32, ()> {
    Err(())
}

#[clippy::msrv = "1.81"]
pub fn returns_unit_error_lint() -> Result<u32, ()> {
    Err(())
}

#[start]
fn main(_argc: isize, _argv: *const *const u8) -> isize {
    0
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[lang = "eh_personality"]
extern "C" fn eh_personality() {}
