#![feature(final_associated_functions)]

trait Foo {
    final fn method() {}
}

impl Foo for () {
    fn method() {}
    //~^ ERROR cannot override `method` because it already has a `final` definition in the trait
}

fn main() {}
