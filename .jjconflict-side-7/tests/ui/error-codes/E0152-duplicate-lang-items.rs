//! Validates the correct printing of E0152 in the case of duplicate "lang_item" function
//! definitions.
//!
//! Issue: <https://github.com/rust-lang/rust/issues/31788>

//@ normalize-stderr: "loaded from \$.*libcore-.*.rmeta" -> "loaded from SYSROOT/libcore-*.rmeta"
//@ dont-require-annotations: NOTE

#![feature(lang_items)]

#[lang = "panic_fmt"]
fn panic_fmt() -> ! {
    //~^ ERROR: found duplicate lang item `panic_fmt`
    //~| NOTE first defined in crate `core`
    loop {}
}

fn main() {}
