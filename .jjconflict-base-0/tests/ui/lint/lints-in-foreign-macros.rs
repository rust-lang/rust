//@ aux-build:lints-in-foreign-macros.rs
//@ check-pass

#![warn(unused_imports)] //~ WARN missing documentation for the crate [missing_docs]
#![warn(missing_docs)]

#[macro_use]
extern crate lints_in_foreign_macros;

macro_rules! foo {
    () => {use std::string::ToString;} //~ WARN: unused import
}

mod a { foo!(); }
mod b { bar!(); }
mod c { baz!(use std::string::ToString;); } //~ WARN: unused import
mod d { baz2!(use std::string::ToString;); } //~ WARN: unused import
baz!(pub fn undocumented() {}); //~ WARN: missing documentation for a function
baz2!(pub fn undocumented2() {}); //~ WARN: missing documentation for a function

fn main() {}
