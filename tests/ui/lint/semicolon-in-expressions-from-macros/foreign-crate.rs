//@ aux-build:foreign-crate.rs
#![deny(semicolon_in_expressions_from_non_local_macros)]

extern crate foreign_crate;

// Test that we fail with a macro in a foreign crate
fn main() {
    let _ = foreign_crate::my_macro!();
    //~^ ERROR trailing semicolon in macro used in expression position
    //~| WARN this was previously accepted
}
