#![feature(final_associated_functions)]

// Regression test for https://github.com/rust-lang/rust/issues/158824.

trait Item {
    final fn bar() -> impl Clone {
        todo!()
    }
}

struct Foo;

impl Item for Foo {
    fn bar() -> impl Clone {}
    //~^ ERROR cannot override `bar` because it already has a `final` definition in the trait
}

fn main() {}
