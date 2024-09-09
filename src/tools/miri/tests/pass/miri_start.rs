//@compile-flags: -Cpanic=abort
#![no_main]
#![no_std]

use core::fmt::Write;

#[path = "../utils/mod.no_std.rs"]
mod utils;

#[no_mangle]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    writeln!(utils::MiriStdout, "Hello from miri_start!").unwrap();
    0
}

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
