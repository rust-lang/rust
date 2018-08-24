// tests the good error message, not "missing module Foo" or something else unexpected

struct Foo;

fn main() {
    Foo::bar();
    //~^ ERROR no function or associated item named `bar` found for type `Foo`
}
