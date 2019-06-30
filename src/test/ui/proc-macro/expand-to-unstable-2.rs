// aux-build:derive-unstable-2.rs

#[macro_use]
extern crate derive_unstable_2;

#[derive(Unstable)]
//~^ ERROR attributes starting with `rustc` are reserved for use by the `rustc` compiler
//~| ERROR attribute `rustc_foo` is currently unknown to the compiler

struct A;

fn main() {
    foo();
}
