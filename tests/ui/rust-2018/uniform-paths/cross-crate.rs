//@ edition:2018
//@ aux-build:cross-crate.rs

extern crate cross_crate;
use cross_crate::*;

#[built_in_attr] //~ ERROR cannot use a built-in attribute through an import
fn main() {
    let _: built_in_type; // OK
}
