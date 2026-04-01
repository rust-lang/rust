//@ aux-build:foreign-crate.rs
//@ check-pass

extern crate foreign_crate;

// Test that we do not lint for a macro in a foreign crate
fn main() {
    let _ = foreign_crate::my_macro!();
}
