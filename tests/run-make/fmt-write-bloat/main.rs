#![feature(lang_items)]
#![feature(start)]
#![no_std]

use core::fmt;
use core::fmt::Write;

#[cfg_attr(not(target_env = "msvc"), link(name = "c"))]
#[cfg_attr(target_env = "msvc", link(name = "libcmt"))]
extern "C" {}

struct Dummy;

impl fmt::Write for Dummy {
    #[inline(never)]
    fn write_str(&mut self, _: &str) -> fmt::Result {
        Ok(())
    }
}

#[start]
fn main(_: isize, _: *const *const u8) -> isize {
    let _ = writeln!(Dummy, "Hello World");
    0
}

#[lang = "eh_personality"]
fn eh_personality() {}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
