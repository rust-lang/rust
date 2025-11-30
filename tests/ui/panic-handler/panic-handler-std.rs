//@ normalize-stderr: "loaded from .*libstd-.*.rmeta" -> "loaded from SYSROOT/libstd-*.rmeta"

extern crate core;

use core::panic::PanicInfo;

#[panic_handler]
fn panic(info: PanicInfo) -> ! { //~ ERROR found duplicate lang item `panic_impl`
    loop {}
}

fn main() {}
