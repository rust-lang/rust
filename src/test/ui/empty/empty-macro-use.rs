// aux-build:two_macros.rs

#[macro_use()]
extern crate two_macros;

pub fn main() {
    macro_two!();
    //~^ ERROR cannot find macro
}
