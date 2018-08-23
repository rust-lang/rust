// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:lints-in-foreign-macros.rs
// compile-pass

#![warn(unused_imports)] //~ missing documentation for crate [missing_docs]
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
