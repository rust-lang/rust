//@ aux-build:derive-bad.rs

#[macro_use]
extern crate derive_bad;

#[derive(A)]
//~^ ERROR proc-macro derive produced unparsable tokens
//~| ERROR expected `:`, found `}`
struct A; //~ ERROR the name `A` is defined multiple times

fn main() {}
