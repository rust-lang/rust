// run-pass
// aux-build:nested-macro-rules.rs
// aux-build:test-macros.rs
// compile-flags: -Z span-debug
// edition:2018

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

extern crate nested_macro_rules;
extern crate test_macros;

use test_macros::print_bang;

use nested_macro_rules::FirstStruct;
struct SecondStruct;

fn main() {
    nested_macro_rules::inner_macro!(print_bang);

    nested_macro_rules::outer_macro!(SecondStruct);
    inner_macro!(print_bang);
}
