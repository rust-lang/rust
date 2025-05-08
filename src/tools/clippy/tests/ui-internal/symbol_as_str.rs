#![feature(rustc_private)]

extern crate rustc_span;

use clippy_utils::sym;
use rustc_span::{Symbol, kw};

fn f(s: Symbol) {
    s.as_str() == "f32";
    //~^ symbol_as_str
    s.as_str() == "proc-macro";
    //~^ symbol_as_str
    s.as_str() == "self";
    //~^ symbol_as_str
    s.as_str() == "msrv";
    //~^ symbol_as_str
    s.as_str() == "Cargo.toml";
    //~^ symbol_as_str
    "get" == s.as_str();
    //~^ symbol_as_str
}
