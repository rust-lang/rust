//! Validates the correct printing of E0152 in the case of duplicate "lang_item" function
//! definitions.
//!
//! Issue: <https://github.com/rust-lang/rust/issues/31788>

//@ normalize-stderr: "loaded from .*libcore-.*.rlib" -> "loaded from SYSROOT/libcore-*.rlib"
//@ dont-require-annotations: NOTE
#![feature(lang_items)]

extern crate core;

use core::panic::PanicInfo;

#[lang = "panic_cannot_unwind"]
fn panic_impl(info: &PanicInfo) -> ! {
    //~^ ERROR: found duplicate lang item `panic_cannot_unwind` [E0152]
    //~| NOTE first defined in crate `core`
    loop {}
}

fn main() {}
