//@ check-pass
//@ compile-flags: -Z span-debug --error-format human
//@ proc-macro: test-macros.rs

#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;

macro_rules! produce_it {
    ($expr:expr) => {
        #[derive(Print)]
        struct Foo(
            [bool; #[cfg_attr(not(FALSE), rustc_dummy(first))] $expr]
        );
    }
}

produce_it!(#[cfg_attr(not(FALSE), rustc_dummy(second))] {
    #![cfg_attr(not(FALSE), rustc_dummy(third))]
    #[cfg_attr(not(FALSE), rustc_dummy(fourth))]
    30
});

fn main() {}
