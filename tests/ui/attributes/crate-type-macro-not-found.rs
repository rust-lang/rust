// Tests for the issue in #137589
#[crate_type = foo!()]
//~^ ERROR cannot find macro `foo` in this scope
//~| ERROR attribute value must be a literal

macro_rules! foo {
    ($x:expr) => {"rlib"}
}

fn main() {}
