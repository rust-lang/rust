//@ compile-flags: -Z span-debug --error-format human
//@ proc-macro: test-macros.rs
//@ check-pass

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;

#[derive(Print)]
struct Foo {
    #[cfg(false)] removed: bool,
    my_array: [bool; {
        struct Inner {
            #[cfg(false)] removed_inner_field: u8,
            non_removed_inner_field: usize
        }
        0
    }]
}

fn main() {}
