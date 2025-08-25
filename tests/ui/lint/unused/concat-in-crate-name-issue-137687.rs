#![deny(unused)]

#[crate_name = concat !()]
//~^ ERROR crate-level attribute should be an inner attribute: add an exclamation mark: `#![foo]
macro_rules! a {
    //~^ ERROR unused macro definition
    () => {};
}
fn main() {}
