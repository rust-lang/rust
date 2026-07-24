//@ normalize-stderr: "loaded from .*libstd-.*.rmeta" -> "loaded from SYSROOT/libstd-*.rmeta"

//~? ERROR multiple implementations of `#[panic_handler]`

extern crate core;

use core::panic::PanicInfo;

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    loop {}
}

fn main() {}
