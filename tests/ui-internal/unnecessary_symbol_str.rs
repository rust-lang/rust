// run-rustfix
#![feature(rustc_private)]
#![deny(clippy::internal)]
#![allow(clippy::unnecessary_operation, unused_must_use)]

extern crate rustc_span;

use rustc_span::symbol::{Ident, Symbol};

fn main() {
    Symbol::intern("foo").as_str() == "clippy";
    Symbol::intern("foo").to_string() == "self";
    Symbol::intern("foo").to_ident_string() != "Self";
    &*Ident::invalid().as_str() == "clippy";
    "clippy" == Ident::invalid().to_string();
}
