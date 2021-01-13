// aux-build:macro_rules.rs

#![warn(clippy::as_conversions)]

#[macro_use]
extern crate macro_rules;

fn with_external_macro() {
    as_conv_with_arg!(0u32 as u64);
    as_conv!();
}

fn main() {
    let i = 0u32 as u64;

    let j = &i as *const u64 as *mut u64;

    with_external_macro();
}
