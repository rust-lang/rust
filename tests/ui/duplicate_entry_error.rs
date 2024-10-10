//@ normalize-stderr-test: "loaded from .*libstd-.*.rmeta" -> "loaded from SYSROOT/libstd-*.rmeta"
// note-pattern: first defined in crate `std`.

// Test for issue #31788 and E0152

#![feature(lang_items)]

extern crate core;

use core::panic::PanicInfo;

#[lang = "panic_impl"]
fn panic_impl(info: &PanicInfo) -> ! {
    //~^ ERROR: found duplicate lang item `panic_impl`
    loop {}
}

fn main() {}
