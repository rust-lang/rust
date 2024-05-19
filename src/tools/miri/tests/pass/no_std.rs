//@compile-flags: -Cpanic=abort
#![feature(start)]
#![no_std]

use core::fmt::Write;

#[path = "../utils/mod.no_std.rs"]
mod utils;

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    writeln!(utils::MiriStdout, "hello, world!").unwrap();
    0
}

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
