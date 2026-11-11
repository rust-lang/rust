//@ compile-flags: -Znext-solver
//@ revisions: fallback constrain
//@[constrain] check-pass

// Tests that we stall the `{integer}: Foo` obligation until after we
// constrain the int type (or fallback occurs).

#![feature(negative_impls, auto_traits)]

auto trait Foo {}

impl !Foo for i32 {}

fn needs_foo(x: impl Foo) {}

fn main() {
    let mut x = 0;
    needs_foo(x);
    //[fallback]~^ ERROR the trait bound `i32: Foo` is not satisfied

    #[cfg(constrain)]
    {
        x = 1u64;
    }
}
