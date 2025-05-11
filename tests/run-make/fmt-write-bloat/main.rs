#![feature(lang_items)]
#![no_main]
#![no_std]

use core::fmt;
use core::fmt::Write;

#[cfg_attr(not(windows), link(name = "c"))]
extern "C" {}

struct Dummy;

impl fmt::Write for Dummy {
    #[inline(never)]
    fn write_str(&mut self, _: &str) -> fmt::Result {
        Ok(())
    }
}

#[no_mangle]
extern "C" fn main(_argc: core::ffi::c_int, _argv: *const *const u8) -> core::ffi::c_int {
    let _ = writeln!(Dummy, "Hello World");
    0
}

#[lang = "eh_personality"]
fn eh_personality() {}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
