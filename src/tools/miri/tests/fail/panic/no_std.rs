//@compile-flags: -Cpanic=abort
#![feature(start, core_intrinsics)]
#![no_std]

use core::fmt::Write;

#[path = "../../utils/mod.no_std.rs"]
mod utils;

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    panic!("blarg I am dead")
}

#[panic_handler]
fn panic_handler(panic_info: &core::panic::PanicInfo) -> ! {
    writeln!(utils::MiriStderr, "{panic_info}").ok();
    core::intrinsics::abort(); //~ ERROR: the program aborted execution
}
