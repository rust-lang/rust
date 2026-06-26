#![deny(unused)]

#[crate_name = concat !()]
//~^ ERROR crate-level attribute should be an inner attribute
macro_rules! a {
    //~^ ERROR unused macro definition
    () => {};
}
fn main() {}
