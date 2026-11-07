//@ aux-build:two_macros.rs

#[macro_use(macro_two, no_way)] //~ ERROR imported macro not found
extern crate two_macros;

pub fn main() {
    macro_two!();
}
