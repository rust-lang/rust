//@ edition: 2024

//! Reject direct const arguments in value/type positions without unrelated brace suggestions.
#![feature(min_generic_const_args)]
#![deny(unused_braces)]

fn main(x: core::direct_const_arg!(2)) {
    //~^ ERROR expected type, found `direct_const_arg!()` constant
    let _ = core::direct_const_arg!(2);
    //~^ ERROR expected expression, found `direct_const_arg!()` constant
    consume({ core::direct_const_arg!(2) });
    //~^ ERROR expected expression, found `direct_const_arg!()` constant
}

fn consume(_: usize) {}
