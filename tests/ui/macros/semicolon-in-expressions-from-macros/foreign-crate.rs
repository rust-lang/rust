//@ aux-build:foreign-crate.rs
//~? ERROR: macro expansion ignores `;` and any tokens following

extern crate foreign_crate;

// Test that we fail with a macro in a foreign crate
fn main() {
    let _ = foreign_crate::my_macro!();
}
