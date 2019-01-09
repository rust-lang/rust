// Test that we use fully-qualified type names in error messages.

mod x {
    pub enum Foo { }
}

mod y {
    pub enum Foo { }
}

fn bar(x: x::Foo) -> y::Foo {
    return x;
    //~^ ERROR mismatched types
    //~| expected type `y::Foo`
    //~| found type `x::Foo`
    //~| expected enum `y::Foo`, found enum `x::Foo`
}

fn main() {
}
