#![allow(clippy::let_unit_value)]
#![feature(rustc_private)]

extern crate rustc_span;

use clippy_utils::sym;
use rustc_span::{Symbol, kw};

fn main() {
    let _ = Symbol::intern("f32");
    //~^ interning_literals

    // Correct suggestion when symbol isn't stringified constant name
    let _ = Symbol::intern("proc-macro");
    //~^ interning_literals

    // Interning a keyword
    let _ = Symbol::intern("self");
    //~^ interning_literals

    // Defined in clippy_utils
    let _ = Symbol::intern("msrv");
    //~^ interning_literals
    let _ = Symbol::intern("Cargo.toml");
    //~^ interning_literals

    // Using a different `intern` function
    let _ = intern("f32");
}

fn intern(_: &str) {}
