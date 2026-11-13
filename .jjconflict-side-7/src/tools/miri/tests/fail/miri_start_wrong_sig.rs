//@compile-flags: -Cpanic=abort
//@error-in-other-file: `miri_start` must have the following signature:
#![no_main]
#![no_std]

use core::fmt::Write;

#[path = "../utils/mod.no_std.rs"]
mod utils;

#[no_mangle]
fn miri_start() -> isize {
    //~^ ERROR: mismatched types
    writeln!(utils::MiriStdout, "Hello from miri_start!").unwrap();
    0
}

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
