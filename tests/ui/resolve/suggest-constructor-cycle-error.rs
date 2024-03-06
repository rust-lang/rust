//@ aux-build:suggest-constructor-cycle-error.rs

// Regression test for https://github.com/rust-lang/rust/issues/119625

extern crate suggest_constructor_cycle_error as a;

const CONST_NAME: a::Uuid = a::Uuid(());
//~^ ERROR: cannot initialize a tuple struct which contains private fields [E0423]

fn main() {}
