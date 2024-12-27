//@ normalize-stderr: "loaded from .*libstd-.*.rlib" -> "loaded from SYSROOT/libstd-*.rlib"
//@ error-pattern: found duplicate lang item `panic_impl`

extern crate core;

use core::panic::PanicInfo;

#[panic_handler]
fn panic(info: PanicInfo) -> ! {
    loop {}
}

fn main() {}
