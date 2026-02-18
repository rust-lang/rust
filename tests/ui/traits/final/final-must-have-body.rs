#![feature(final_associated_functions)]

trait Foo {
    final fn method();
    //~^ ERROR `final` is only allowed on associated functions if they have a body
}

fn main() {}
