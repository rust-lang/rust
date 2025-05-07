#![feature(lang_items, libc)]
#![no_std]
#![no_main]
#![warn(clippy::result_unit_err)]

#[clippy::msrv = "1.80"]
pub fn returns_unit_error_no_lint() -> Result<u32, ()> {
    Err(())
}

#[clippy::msrv = "1.81"]
pub fn returns_unit_error_lint() -> Result<u32, ()> {
    //~^ result_unit_err
    Err(())
}

#[unsafe(no_mangle)]
extern "C" fn main(_argc: core::ffi::c_int, _argv: *const *const u8) -> core::ffi::c_int {
    0
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[lang = "eh_personality"]
extern "C" fn eh_personality() {}
