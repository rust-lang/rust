// Macro attributes are allowed after `#[derive]` and
// `#[derive]` fully configures the item for following attributes.

//@ check-pass
//@ compile-flags: -Z span-debug
//@ proc-macro: test-macros.rs

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;

#[print_attr]
#[derive(Print)]
struct AttributeDerive {
    #[cfg(false)]
    field: u8,
}

#[derive(Print)]
#[print_attr]
struct DeriveAttribute {
    #[cfg(false)]
    field: u8,
}

fn main() {}
