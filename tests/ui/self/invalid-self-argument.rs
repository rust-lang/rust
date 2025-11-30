//! regression test for <https://github.com/rust-lang/rust/issues/55972>

fn a(&self) {}
//~^ ERROR `self` parameter is only allowed in associated functions
//~| NOTE not semantically valid as function parameter
//~| NOTE associated functions are those in `impl` or `trait` definitions

fn b(foo: u32, &mut self) {}
//~^ ERROR unexpected `self` parameter in function
//~| NOTE must be the first parameter of an associated function

struct Foo {}

impl Foo {
    fn c(foo: u32, self) {}
    //~^ ERROR unexpected `self` parameter in function
    //~| NOTE must be the first parameter of an associated function

    fn good(&mut self, foo: u32) {}
}

fn main() {}
