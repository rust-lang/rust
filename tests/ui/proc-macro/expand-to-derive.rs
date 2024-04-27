//@ check-pass
//@ compile-flags: -Z span-debug --error-format human
//@ aux-build:test-macros.rs

#![feature(rustc_attrs)]

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;

macro_rules! expand_to_derive {
    ($item:item) => {
        #[derive(Print)]
        struct Foo {
            #[cfg(FALSE)] removed: bool,
            field: [bool; {
                $item
                0
            }]
        }
    };
}

expand_to_derive! {
    #[cfg_attr(not(FALSE), rustc_dummy)]
    struct Inner {
        #[cfg(FALSE)] removed_inner_field: bool,
        other_inner_field: u8,
    }
}

fn main() {}
