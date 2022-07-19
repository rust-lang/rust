// check-pass
// edition:2018
// compile-flags: -Z span-debug
// aux-build:test-macros.rs
// aux-build:dollar-dollar-crate-external.rs

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;
extern crate dollar_dollar_crate_external;

struct S;

mod namespace {
    use dollar_dollar_crate_external::define_macro;

    define_macro!(m => S);

    #[print_attr]
    const _: crate::S = m!();
}

fn main() {}
