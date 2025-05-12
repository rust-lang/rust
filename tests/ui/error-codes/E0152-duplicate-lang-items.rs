//! Validates the correct printing of E0152 in the case of duplicate "lang_item" function
//! definitions.
//!
//! Issue: <https://github.com/rust-lang/rust/issues/31788>

//@ normalize-stderr: "loaded from .*libstd-.*.rlib" -> "loaded from SYSROOT/libstd-*.rlib"
//@ dont-require-annotations: NOTE

#![feature(lang_items)]

extern crate core;

use core::panic::PanicInfo;

#[lang = "panic_impl"]
fn panic_impl(info: &PanicInfo) -> ! {
    //~^ ERROR: found duplicate lang item `panic_impl`
    //~| NOTE first defined in crate `std`
    loop {}
}

fn main() {}
