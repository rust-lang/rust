// aux-build:derive-bad.rs

#[macro_use]
extern crate derive_bad;

#[derive(
    A
)]
//~^^ ERROR proc-macro derive produced unparseable tokens
//~| ERROR expected `:`, found `}`
struct A;

fn main() {}
