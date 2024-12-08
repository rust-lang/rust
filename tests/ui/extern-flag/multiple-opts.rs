//@ aux-crate:priv,noprelude:somedep=somedep.rs
//@ compile-flags: -Zunstable-options
//@ edition:2018

// Test for multiple options to --extern. Can't test for errors from both
// options at the same time, so this only checks that noprelude is honored.

#![warn(exported_private_dependencies)]

// Module to avoid adding to prelude.
pub mod m {
    extern crate somedep;
    pub struct PublicType {
        pub field: somedep::S,
    }
}

fn main() {
    somedep::somefun();  //~ ERROR failed to resolve
}
