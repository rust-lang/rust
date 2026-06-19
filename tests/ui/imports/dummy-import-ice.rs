// Regression test for issue #157406.

//@ check-pass
//@ proc-macro: dummy-import-ice-macro.rs

extern crate dummy_import_ice_macro;

pub fn foo() {
    ambiguous();
}

mod submodule {
    pub fn ambiguous() {}
}

pub mod ambiguous {}

dummy_import_ice_macro::my_macro!();

fn main() {}
