//@ normalize-stderr-test: "loaded from .*libstd-.*.rmeta" -> "loaded from SYSROOT/libstd-*.rmeta"
//@ error-pattern: found duplicate lang item `panic_impl`

extern crate core;

use core::panic::PanicInfo;

#[panic_handler]
fn panic(info: PanicInfo) -> ! {
    loop {}
}

fn main() {}
