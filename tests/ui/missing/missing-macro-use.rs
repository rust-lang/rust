//@ aux-build:two_macros.rs

extern crate two_macros;

pub fn main() {
    macro_two!();
    //~^ ERROR cannot find macro `macro_two` in this scope
}
