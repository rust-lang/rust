//! Auxiliary crate testing this issue https://github.com/rust-lang/rust/issues/32518
//@ no-prefer-dynamic
//@ compile-flags: -Ccodegen-units=2 --crate-type=lib

extern crate no_duplicate_symbols_with_codegen_units_cgu_test as cgu_test;

pub mod a {
    pub fn a() {
        ::cgu_test::id(0);
    }
}
pub mod b {
    pub fn a() {
        ::cgu_test::id(0);
    }
}
