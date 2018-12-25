// note-pattern: first defined in crate `std`.

// Test for issue #31788 and E0152

#![feature(lang_items)]

use std::panic::PanicInfo;

#[lang = "panic_impl"]
fn panic_impl(info: &PanicInfo) -> ! {
//~^ ERROR: duplicate lang item found: `panic_impl`.
    loop {}
}

fn main() {}
