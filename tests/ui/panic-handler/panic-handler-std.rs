//@ normalize-stderr: "loaded from .*libcore-.*.rlib" -> "loaded from SYSROOT/libcore-*.rlib"

extern crate core;

use core::panic::PanicInfo;

#[panic_handler]
fn panic(info: PanicInfo) -> ! { //~ ERROR found duplicate lang item `panic_impl`
    loop {}
}

fn main() {}
