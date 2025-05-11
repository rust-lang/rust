// Test that we use fully-qualified type names in error messages.

//@ dont-require-annotations: NOTE

mod x {
    pub enum Foo { }
}

mod y {
    pub enum Foo { }
}

fn bar(x: x::Foo) -> y::Foo {
    return x;
    //~^ ERROR mismatched types
    //~| NOTE expected `y::Foo`, found `x::Foo`
}

fn main() {
}
