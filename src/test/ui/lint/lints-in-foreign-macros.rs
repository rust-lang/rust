// aux-build:lints-in-foreign-macros.rs
// compile-pass

#![warn(unused_imports)]

#[macro_use]
extern crate lints_in_foreign_macros;

macro_rules! foo {
    () => {use std::string::ToString;} //~ WARN: unused import
}

mod a { foo!(); }
mod b { bar!(); }
mod c { baz!(use std::string::ToString;); } //~ WARN: unused import
mod d { baz2!(use std::string::ToString;); } //~ WARN: unused import

fn main() {}
