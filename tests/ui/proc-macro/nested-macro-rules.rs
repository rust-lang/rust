//@ run-pass
//@ aux-build:nested-macro-rules.rs
//@ proc-macro: test-macros.rs
//@ compile-flags: -Z span-debug -Z macro-backtrace
//@ edition:2018

#![no_std] // Don't load unnecessary hygiene information from std
#![warn(non_local_definitions)]

extern crate std;

extern crate nested_macro_rules;
extern crate test_macros;

use test_macros::{print_bang, print_attr};

use nested_macro_rules::FirstStruct;
struct SecondStruct;

fn main() {
    nested_macro_rules::inner_macro!(print_bang, print_attr);

    nested_macro_rules::outer_macro!(SecondStruct, SecondAttrStruct);
    //~^ WARN non-local `macro_rules!` definition
    inner_macro!(print_bang, print_attr);
}
