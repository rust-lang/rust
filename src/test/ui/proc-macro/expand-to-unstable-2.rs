// aux-build:derive-unstable-2.rs

#[macro_use]
extern crate derive_unstable_2;

#[derive(Unstable)]
//~^ ERROR attribute `rustc_foo` is currently unknown
struct A;

fn main() {
    foo();
}
