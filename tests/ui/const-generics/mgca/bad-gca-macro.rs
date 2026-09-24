//@ edition: 2024

//! Reject direct const arguments in value/type positions without unrelated brace suggestions.
#![feature(min_generic_const_args)]
#![deny(unused_braces)]

use std::gca;

fn main(x: gca!(2)) {
    //~^ ERROR expected type, found `gca!()` constant
    let _ = gca!(2);
    //~^ ERROR expected expression, found `gca!()` constant
    consume({ gca!(2) });
    //~^ ERROR expected expression, found `gca!()` constant
}

fn consume(_: usize) {}
