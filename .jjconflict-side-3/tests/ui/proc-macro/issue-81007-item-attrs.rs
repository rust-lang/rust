//@ check-pass
//@ edition:2018
//@ compile-flags: -Z span-debug
//@ proc-macro: test-macros.rs

#![feature(rustc_attrs)]

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use] extern crate test_macros;

macro_rules! capture_item {
    ($item:item) => {
        #[print_attr]
        $item
    }
}

capture_item! {
    /// A doc comment
    struct Foo {}
}

capture_item! {
    #[rustc_dummy]
    /// Another comment comment
    struct Bar {}
}

fn main() {}
