#![feature(rustc_private)]
#![deny(clippy::internal)]
#![allow(
    clippy::slow_symbol_comparisons,
    clippy::borrow_deref_ref,
    clippy::unnecessary_operation,
    unused_must_use,
    clippy::missing_clippy_version_attribute
)]

extern crate rustc_span;

use rustc_span::symbol::{Ident, Symbol};

fn main() {
    Symbol::intern("foo").as_str() == "clippy";
    //~^ unnecessary_symbol_str
    Symbol::intern("foo").to_string() == "self";
    //~^ unnecessary_symbol_str
    Symbol::intern("foo").to_ident_string() != "Self";
    //~^ unnecessary_symbol_str
    &*Ident::empty().as_str() == "clippy";
    //~^ unnecessary_symbol_str
    "clippy" == Ident::empty().to_string();
    //~^ unnecessary_symbol_str
}
