//@ normalize-stderr: "loaded from .*libstd-.*.rlib" -> "loaded from SYSROOT/libstd-*.rlib"

extern crate core;

use core::panic::PanicInfo;

#[panic_handler]
fn panic(info: PanicInfo) -> ! { //~ ERROR found duplicate lang item `panic_impl`
    loop {}
}

fn main() {}
