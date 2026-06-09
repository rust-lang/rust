// tests the good error message, not "missing module Foo" or something else unexpected

struct Foo;

fn main() {
    Foo::bar();
    //~^ ERROR no associated function or constant named `bar` found for struct `Foo`
}
