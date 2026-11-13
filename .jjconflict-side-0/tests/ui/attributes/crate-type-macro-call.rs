#![crate_type = foo!()]
//~^ ERROR attribute value must be a literal

macro_rules! foo {
    () => {"rlib"};
}

fn main() {}
